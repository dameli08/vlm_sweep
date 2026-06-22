#!/usr/bin/env python3
import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from gemini_answer_extractor import extract_final_answer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--answers-root", required=True)
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--sweep-param", required=True)
    parser.add_argument("--sweep-value", required=True)
    parser.add_argument("--temperature", required=True)
    parser.add_argument("--top-p", required=True)
    parser.add_argument("--top-k", required=True)
    parser.add_argument("--repetition-penalty", required=True)
    parser.add_argument("--gemini-api-key", default="")
    parser.add_argument("--gemini-model", default="gemini-3.1-flash-lite")
    return parser.parse_args()


def to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def split_reasoning(text):
    text = to_text(text)
    paired = re.search(r"<think>(.*?)</think>(.*)$", text, flags=re.I | re.S)
    if paired:
        return paired.group(1).strip(), paired.group(2).strip()
    closing = list(re.finditer(r"</think>", text, flags=re.I))
    if closing:
        marker = closing[-1]
        thinking = re.sub(r"^\s*<think>", "", text[: marker.start()], flags=re.I).strip()
        return thinking, text[marker.end() :].strip()
    analysis = re.search(r"<analysis>(.*?)</analysis>(.*)$", text, flags=re.I | re.S)
    if analysis:
        return analysis.group(1).strip(), analysis.group(2).strip()
    return "", text.strip()


def strip_think_blocks(text):
    return split_reasoning(text)[1]


def normalize_text(text):
    text = strip_think_blocks(text)
    try:
        import ast

        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            text = " ".join(str(item) for item in parsed)
        elif isinstance(parsed, tuple):
            text = " ".join(str(item) for item in parsed)
    except Exception:
        pass
    text = re.sub(r"[\[\]\(\)\{\}\"'`$]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def extract_letter(text):
    cleaned = strip_think_blocks(text).upper()
    patterns = [
        r"FINAL\s*ANSWER\s*(?:IS|:|=|-)?\s*\(?\s*([A-Z])\s*\)?",
        r"ANSWER\s*(?:IS|:|=|-)?\s*\(?\s*([A-Z])\s*\)?",
        r"OPTION\s*(?:IS|:|=|-)?\s*\(?\s*([A-Z])\s*\)?",
        r"CHOICE\s*(?:IS|:|=|-)?\s*\(?\s*([A-Z])\s*\)?",
    ]
    for pattern in patterns:
        hits = re.findall(pattern, cleaned)
        if hits:
            return hits[-1]
    solo = re.fullmatch(r"\(?\s*([A-Z])\s*[\)\].,:;\-!?]*\s*", cleaned)
    if solo:
        return solo.group(1)
    tail_tokens = re.findall(r"\b([A-Z])\b", cleaned[-80:])
    return tail_tokens[-1] if tail_tokens else ""


def benchmark_key(name):
    lower = name.lower()
    if "mmmu" in lower:
        return "mmmu_pro"
    if "ocrbench" in lower:
        return "ocrbench"
    if "mmstar" in lower:
        return "mmstar"
    return re.sub(r"[^a-z0-9]+", "_", lower).strip("_")


def benchmark_rule(name):
    key = benchmark_key(name)
    if key in {"mmmu_pro", "mmstar"}:
        return "letter"
    if key == "ocrbench":
        return "gemini_judge"
    return "letter"


def pick_prediction(record):
    for key in ("prediction", "pred", "response", "resps", "filtered_resps", "model_output", "model_outputs"):
        if key in record and record[key] is not None:
            return record[key]
    return None


def pick_target(record):
    if "target" in record:
        return record["target"]
    doc = record.get("doc")
    if isinstance(doc, dict):
        for key in ("answer", "target", "label"):
            if key in doc:
                return doc[key]
    return None


def pick_doc_id(record):
    for key in ("doc_id", "id", "instance_id"):
        if key in record:
            return record[key]
    doc = record.get("doc")
    if isinstance(doc, dict):
        for key in ("id", "question_id", "pid"):
            if key in doc:
                return doc[key]
    return ""


def load_live_rows(out_dir, benchmark):
    live_rows = {}
    live_csv = out_dir / "answers_live.csv"
    if not live_csv.exists():
        return live_rows
    try:
        with live_csv.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = str(row.get("doc_id", ""))
                live_rows[key] = row
                live_rows[(benchmark, key)] = row
    except Exception:
        return {}
    return live_rows


def iter_sample_records(out_dir):
    candidates = []
    for path in out_dir.rglob("*"):
        if not path.is_file():
            continue
        if "sample" not in path.name.lower():
            continue
        if path.suffix.lower() in {".json", ".jsonl"}:
            candidates.append(path)

    for path in sorted(candidates):
        if path.suffix.lower() == ".jsonl":
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                if isinstance(record, dict):
                    yield record
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and isinstance(payload.get("samples"), list):
            records = payload["samples"]
        else:
            records = []
        for record in records:
            if isinstance(record, dict):
                yield record


def collect_rows(out_dir, benchmark):
    live_rows = load_live_rows(out_dir, benchmark)
    rows = []

    for record in iter_sample_records(out_dir):
        doc_id = str(pick_doc_id(record))
        live = live_rows.get((benchmark, doc_id)) or live_rows.get(doc_id) or {}
        target = to_text(pick_target(record))
        raw_prediction = to_text(pick_prediction(record))
        live_raw = to_text(live.get("raw_response_full", ""))
        source_response = live_raw or raw_prediction
        parsed_thinking, parsed_response = split_reasoning(source_response)
        visible = to_text(live.get("visible_response", "")) or parsed_response
        if "</think>" in visible.lower() or "<think>" in visible.lower():
            fallback_thinking, visible = split_reasoning(visible)
            parsed_thinking = parsed_thinking or fallback_thinking
        rows.append(
            {
                "doc_id": doc_id,
                "true_answer": target,
                "response": visible.strip(),
                "thinking_process": to_text(live.get("thinking_trace", "")) or parsed_thinking,
            }
        )

    if not rows and live_rows:
        seen = set()
        for key, live in live_rows.items():
            if isinstance(key, tuple):
                continue
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "doc_id": key,
                    "true_answer": to_text(live.get("true_answer_text", "")),
                    "response": to_text(live.get("visible_response") or live.get("prediction_clean") or live.get("prediction_raw", "")),
                    "thinking_process": to_text(live.get("thinking_trace", "")),
                }
            )

    return rows


def gemini_judge(api_key, model, true_answer, response):
    if not api_key:
        true_norm = normalize_text(true_answer)
        response_norm = normalize_text(response)
        return bool(true_norm and response_norm and (true_norm == response_norm or true_norm in response_norm)), "heuristic_no_gemini_key"

    prompt = (
        "You are judging an OCR benchmark answer. Determine whether the model response "
        "contains the same final answer as the ground truth. Ignore harmless wording, "
        "quotes, currency/math symbols, punctuation, and phrases such as 'the answer is'. "
        "Return only JSON with this schema: {\"correct\": true or false}.\n\n"
        f"Ground truth: {true_answer}\n"
        f"Model response: {response}\n"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as result:
            data = json.loads(result.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        true_norm = normalize_text(true_answer)
        response_norm = normalize_text(response)
        fallback = bool(true_norm and response_norm and (true_norm == response_norm or true_norm in response_norm))
        return fallback, f"heuristic_after_gemini_error:{exc}"

    text = ""
    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            text += part.get("text", "")
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            judged = json.loads(match.group(0))
            return bool(judged.get("correct")), "gemini"
        except json.JSONDecodeError:
            pass
    return ("true" in text.lower() and "false" not in text.lower()), "gemini_text_fallback"


def evaluate(rows, benchmark, gemini_api_key, gemini_model):
    rule = benchmark_rule(benchmark)
    total = len(rows)
    valid = 0
    correct = 0
    invalid = 0
    judge_notes = {}

    for row in rows:
        true_answer = row["true_answer"]
        response = row["response"]
        if not normalize_text(true_answer) or not normalize_text(response):
            invalid += 1
            continue

        if rule == "letter":
            true_letter = extract_letter(true_answer)
            pred_letter = extract_letter(response)
            if not true_letter or not pred_letter:
                invalid += 1
                continue
            valid += 1
            if true_letter == pred_letter:
                correct += 1
            continue

        valid += 1
        is_correct, judge = gemini_judge(gemini_api_key, gemini_model, true_answer, response)
        judge_notes[judge] = judge_notes.get(judge, 0) + 1
        if is_correct:
            correct += 1

    accuracy = correct / valid if valid else None
    return {
        "accuracy": accuracy,
        "correct": correct,
        "valid_count": valid,
        "invalid_count": invalid,
        "total_rows": total,
        "rule": rule,
        "judge_notes": judge_notes,
    }


def answer_file_path(answers_root, model_name, benchmark, mode, sweep_param, sweep_value):
    model_dir = answers_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    key = benchmark_key(benchmark)
    safe_value = re.sub(r"[^A-Za-z0-9_.-]+", "_", sweep_value)
    return model_dir / f"{key}__{mode}__{sweep_param}_{safe_value}.jsonl"


def load_answer_file(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            rows.append(
                {
                    "true_answer": to_text(record.get("true_answer", "")),
                    "response": to_text(record.get("response", "")),
                    "thinking_process": to_text(record.get("thinking_process", "")),
                }
            )
    return rows



def complete_reason_generation(row):
    thinking = to_text(row.get("thinking_process", "")).strip()
    response = to_text(row.get("response", "")).strip()
    if not response or response in thinking[-max(2000, len(response) + 100):]:
        return thinking or response
    return f"{thinking}\n{response}".strip()


def normalize_reason_rows_with_gemini(rows, args, only_empty=False):
    if not args.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY is required to normalize reason-mode answers")
    normalized = []
    attempted = 0
    for row in rows:
        generation = complete_reason_generation(row)
        current_response = to_text(row.get("response", "")).strip()
        if only_empty and current_response:
            final_answer = current_response
        else:
            final_answer = extract_final_answer(
                args.gemini_api_key,
                args.gemini_model,
                args.benchmark,
                generation,
            )
            attempted += 1
            if attempted % 50 == 0:
                print(f"[INFO] Gemini-normalized {attempted} reason answers")
        normalized.append(
            {
                "true_answer": row.get("true_answer", ""),
                "thinking_process": generation,
                "response": final_answer,
            }
        )
    return normalized

def write_answer_file(rows, answers_root, model_name, benchmark, mode, sweep_param, sweep_value):
    path = answer_file_path(answers_root, model_name, benchmark, mode, sweep_param, sweep_value)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if mode == "reason":
                record = {
                    "true_answer": row["true_answer"],
                    "thinking_process": row.get("thinking_process", ""),
                    "response": row["response"],
                }
            else:
                record = {"true_answer": row["true_answer"], "response": row["response"]}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def append_result(args, metric):
    results_path = Path(args.results_jsonl)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    key = benchmark_key(args.benchmark)
    record = {
        "model_name": args.model_name,
        "mode": args.mode,
        "run_name": f"{key}_{args.sweep_param}_{args.sweep_value}",
        "parameters": {
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(float(args.top_k)),
            "repetition_penalty": float(args.repetition_penalty),
        },
        "benchmark": key,
        "accuracy": metric["accuracy"],
    }
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    answers_root = Path(args.answers_root)
    answer_path = answer_file_path(
        answers_root,
        args.model_name,
        args.benchmark,
        args.mode,
        args.sweep_param,
        args.sweep_value,
    )
    if args.mode == "reason":
        rows = load_answer_file(answer_path)
        if not rows:
            raise SystemExit(f"Reason-mode live answer file is empty or missing: {answer_path}")
        ready_path = out_dir / ".gemini_answers_ready"
        deadline = time.time() + 7200
        while not ready_path.exists() and (out_dir / ".live_writer_stop").exists():
            if time.time() > deadline:
                raise SystemExit(f"Timed out waiting for Gemini answer extraction: {answer_path}")
            time.sleep(1)
        rows = load_answer_file(answer_path)
        if not ready_path.exists():
            # Compatibility fallback for an older writer that did not perform
            # Gemini extraction. Current writers leave failed responses empty.
            rows = normalize_reason_rows_with_gemini(rows, args)
            answer_path = write_answer_file(
                rows,
                answers_root,
                args.model_name,
                args.benchmark,
                args.mode,
                args.sweep_param,
                args.sweep_value,
            )
            ready_path.touch()
    else:
        rows = collect_rows(out_dir, args.benchmark)
        answer_path = write_answer_file(rows, answers_root, args.model_name, args.benchmark, args.mode, args.sweep_param, args.sweep_value)
    metric = evaluate(rows, args.benchmark, args.gemini_api_key, args.gemini_model)
    append_result(args, metric)
    print(f"[INFO] wrote answers={answer_path} accuracy={metric['accuracy']} valid={metric['valid_count']} invalid={metric['invalid_count']}")


if __name__ == "__main__":
    main()
