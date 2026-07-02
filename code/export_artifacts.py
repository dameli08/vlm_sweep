#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import re
import time
from pathlib import Path

from answer_extraction import extract_final_letter_strict, split_reasoning_response, strip_reasoning_blocks


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
    return parser.parse_args()


def to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def normalize_text(text):
    text = strip_reasoning_blocks(text)
    try:
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


def benchmark_key(name):
    lower = (name or "").lower()
    if "mmmu" in lower:
        return "mmmu_pro"
    if "ai2d" in lower:
        return "ai2d"
    return re.sub(r"[^a-z0-9]+", "_", lower).strip("_")


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


def pick_doc(record):
    doc = record.get("doc") if isinstance(record, dict) else None
    return doc if isinstance(doc, dict) else {}


def option_letter(index):
    try:
        index = int(index)
    except (TypeError, ValueError):
        return ""
    if 0 <= index < 26:
        return chr(ord("A") + index)
    return ""


def extract_true_letter(text):
    text = to_text(text).strip()
    digit = option_letter(text)
    if digit:
        return digit
    labelled = re.match(r"^\s*([A-J])\s*[\.)\]:-]", text, flags=re.I)
    if labelled:
        return labelled.group(1).upper()
    solo = re.fullmatch(r"\s*([A-J])\s*", text, flags=re.I)
    if solo:
        return solo.group(1).upper()
    return extract_final_letter_strict(text)


def canonical_true_answer(benchmark, target, record=None):
    target_text = to_text(target).strip()
    if benchmark_key(benchmark) != "ai2d":
        return target_text

    doc = pick_doc(record or {})
    options = doc.get("options")
    answer = doc.get("answer", target)
    if isinstance(options, (list, tuple)) and options:
        idx = None
        try:
            idx = int(answer)
        except (TypeError, ValueError):
            answer_text = to_text(answer).strip().upper()
            if re.fullmatch(r"[A-Z]", answer_text):
                idx = ord(answer_text) - ord("A")
        if idx is None:
            try:
                idx = int(target)
            except (TypeError, ValueError):
                target_letter = extract_true_letter(target_text)
                if target_letter:
                    idx = ord(target_letter) - ord("A")
        if idx is not None and 0 <= idx < len(options):
            return f"{chr(ord('A') + idx)}. {to_text(options[idx]).strip()}"

    target_letter = option_letter(target_text)
    if target_letter:
        return target_letter
    return target_text


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
        target = canonical_true_answer(benchmark, pick_target(record), record)
        raw_prediction = to_text(pick_prediction(record))
        live_raw = to_text(live.get("raw_response_full", ""))
        source_response = live_raw or raw_prediction
        parsed_thinking, parsed_response = split_reasoning_response(source_response)
        visible = to_text(live.get("visible_response", "")) or parsed_response
        rows.append(
            {
                "doc_id": doc_id,
                "true_answer": target,
                "response": extract_final_letter_strict(visible),
                "thinking_process": to_text(live.get("thinking_trace", "")) or parsed_thinking,
            }
        )

    if not rows and live_rows:
        seen = set()
        for key, live in live_rows.items():
            if isinstance(key, tuple) or key in seen:
                continue
            seen.add(key)
            response = to_text(live.get("visible_response") or live.get("prediction_clean") or live.get("prediction_raw", ""))
            rows.append(
                {
                    "doc_id": key,
                    "true_answer": canonical_true_answer(benchmark, live.get("true_answer_text", "")),
                    "response": extract_final_letter_strict(response),
                    "thinking_process": to_text(live.get("thinking_trace", "")),
                }
            )

    return rows


def evaluate(rows, benchmark):
    total = len(rows)
    valid = 0
    correct = 0
    invalid = 0

    for row in rows:
        true_answer = row.get("true_answer", "")
        response = row.get("response", "")
        true_letter = extract_true_letter(true_answer)
        pred_letter = extract_final_letter_strict(response)
        if not true_letter or not pred_letter:
            invalid += 1
            continue
        valid += 1
        if true_letter == pred_letter:
            correct += 1

    return {
        "accuracy": correct / valid if valid else None,
        "correct": correct,
        "valid_count": valid,
        "invalid_count": invalid,
        "total_rows": total,
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
        "overall_questions": metric["total_rows"],
        "answered_questions": metric["valid_count"],
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
        ready_path = out_dir / ".answers_ready"
        deadline = time.time() + 7200
        while not ready_path.exists() and (out_dir / ".live_writer_stop").exists():
            if time.time() > deadline:
                raise SystemExit(f"Timed out waiting for answer extraction: {answer_path}")
            time.sleep(1)
        rows = load_answer_file(answer_path)
    else:
        rows = collect_rows(out_dir, args.benchmark)
        answer_path = write_answer_file(rows, answers_root, args.model_name, args.benchmark, args.mode, args.sweep_param, args.sweep_value)
    metric = evaluate(rows, args.benchmark)
    append_result(args, metric)
    print(f"[INFO] wrote answers={answer_path} accuracy={metric['accuracy']} valid={metric['valid_count']} invalid={metric['invalid_count']}")


if __name__ == "__main__":
    main()
