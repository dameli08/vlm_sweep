#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import math
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from answer_extraction import (
    clean_final_answer_text,
    extract_benchmark_answer,
    extract_final_letter_strict,
    split_reasoning_response,
    strip_reasoning_blocks,
)

try:
    from lmms_eval.tasks.mathvision.eval_utils import find_math_answer, is_equal as mathvision_is_equal
except Exception:
    find_math_answer = None
    mathvision_is_equal = None


GENERATION_METADATA_FIELDS = (
    "finish_reason",
    "budget_truncation_flag",
    "trace_token_count",
    "answer_token_count",
    "loop_score",
    "max_tokens",
    "thinking_budget",
    "model_id",
    "model_path",
    "vllm_version",
    "harness_git_commit",
    "timestamp",
    "request_success",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
)


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
    parser.add_argument("--presence-penalty", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--sample-seeds", default="")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--max-tokens", default="")
    parser.add_argument("--thinking-budget", default="")
    parser.add_argument("--vllm-version", default="")
    parser.add_argument("--harness-git-commit", default="")
    parser.add_argument("--subset-path", default="")
    return parser.parse_args()


def to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def benchmark_key(name):
    lower = (name or "").lower()
    if "mmmu" in lower:
        return "mmmu_pro"
    if "ai2d" in lower:
        return "ai2d"
    if "mathvision" in lower:
        return "mathvision_testmini"
    return re.sub(r"[^a-z0-9]+", "_", lower).strip("_")


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



def true_answer_is_explicit_letter(text):
    text = to_text(text).strip()
    return bool(
        re.match(r"^\s*([A-J])\s*[\.)\]:-]", text, flags=re.I)
        or re.fullmatch(r"\s*([A-J])\s*", text, flags=re.I)
    )

def normalize_text(text):
    text = strip_reasoning_blocks(to_text(text))
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


def normalize_mathvision_answer(text):
    text = clean_final_answer_text(text)
    if find_math_answer is not None:
        try:
            text = find_math_answer(text)
        except Exception:
            pass
    return (
        text.replace("(a)", "a")
        .replace("(b)", "b")
        .replace("(c)", "c")
        .replace("(d)", "d")
        .replace("(e)", "e")
        .replace("{a}", "a")
        .replace("{b}", "b")
        .replace("{c}", "c")
        .replace("{d}", "d")
        .replace("{e}", "e")
        .rstrip(".")
        .lstrip(":")
        .strip()
    )


def score_row(row, benchmark):
    true_answer = to_text(row.get("true_answer", ""))
    response = to_text(row.get("response", ""))
    key = benchmark_key(benchmark)
    true_letter = extract_true_letter(true_answer)

    if key in {"mmmu_pro", "ai2d"}:
        pred_letter = extract_final_letter_strict(response)
        if not pred_letter:
            return 0, True, "letter"
        return (1 if pred_letter == true_letter else 0), False, "letter"

    if key == "mathvision_testmini" and true_answer_is_explicit_letter(true_answer):
        pred_letter = extract_final_letter_strict(response)
        if not pred_letter:
            return 0, True, "letter"
        return (1 if pred_letter == true_letter else 0), False, "letter"

    pred = normalize_mathvision_answer(response)
    gold = normalize_mathvision_answer(true_answer)
    if not pred:
        return 0, True, "math_equivalence"
    if mathvision_is_equal is not None:
        try:
            return (1 if mathvision_is_equal(gold, pred) else 0), False, "math_equivalence"
        except Exception:
            pass
    return (1 if normalize_text(gold) == normalize_text(pred) else 0), False, "normalized_exact"


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


def pick_doc(record):
    doc = record.get("doc") if isinstance(record, dict) else None
    return doc if isinstance(doc, dict) else {}


def pick_doc_id(record):
    for key in ("doc_id", "id", "instance_id"):
        if key in record:
            return record[key]
    doc = pick_doc(record)
    for key in ("id", "question_id", "pid"):
        if key in doc:
            return doc[key]
    return ""


def canonical_true_answer(benchmark, target, record=None):
    key = benchmark_key(benchmark)
    target_text = to_text(target).strip()
    doc = pick_doc(record or {})
    options = doc.get("options")
    answer = doc.get("answer", target)

    if key in {"ai2d", "mathvision_testmini"} and isinstance(options, (list, tuple)) and options:
        idx = None
        try:
            idx = int(answer)
        except (TypeError, ValueError):
            answer_text = to_text(answer).strip().upper()
            if re.fullmatch(r"[A-Z]", answer_text):
                idx = ord(answer_text) - ord("A")
        if idx is None:
            letter = extract_true_letter(target_text)
            if letter:
                idx = ord(letter) - ord("A")
        if idx is not None and 0 <= idx < len(options):
            return f"{chr(ord('A') + idx)}. {to_text(options[idx]).strip()}"

    if key == "ai2d":
        letter = option_letter(target_text)
        if letter:
            return letter
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
        if path.is_file() and "sample" in path.name.lower() and path.suffix.lower() in {".json", ".jsonl"}:
            candidates.append(path)
    for path in sorted(candidates):
        if path.suffix.lower() == ".jsonl":
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in lines:
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
        records = payload if isinstance(payload, list) else payload.get("samples", []) if isinstance(payload, dict) else []
        for record in records:
            if isinstance(record, dict):
                yield record


def collect_rows(out_dir, benchmark, mode, seed):
    live_rows = load_live_rows(out_dir, benchmark)
    rows = []
    for record in iter_sample_records(out_dir):
        item_id = str(pick_doc_id(record))
        live = live_rows.get((benchmark, item_id)) or live_rows.get(item_id) or {}
        target = canonical_true_answer(benchmark, pick_target(record), record)
        raw_prediction = to_text(pick_prediction(record))
        live_raw = to_text(live.get("raw_response_full", ""))
        source_response = live_raw or raw_prediction
        thinking, parsed_response = split_reasoning_response(source_response)
        visible = to_text(live.get("visible_response", "")) or parsed_response
        rows.append(
            {
                "item_id": item_id,
                "seed": seed,
                "true_answer": target,
                "response": extract_benchmark_answer(visible, benchmark, target),
                "thinking_process": to_text(live.get("thinking_trace", "")) or thinking,
                "raw_output": source_response,
                **{field: live.get(field, "") for field in GENERATION_METADATA_FIELDS},
            }
        )
    if not rows and live_rows:
        seen = set()
        for key, live in live_rows.items():
            if isinstance(key, tuple) or key in seen:
                continue
            seen.add(key)
            target = canonical_true_answer(benchmark, live.get("true_answer_text", ""))
            response = to_text(live.get("visible_response") or live.get("prediction_clean") or live.get("prediction_raw", ""))
            rows.append(
                {
                    "item_id": key,
                    "seed": seed,
                    "true_answer": target,
                    "response": extract_benchmark_answer(response, benchmark, target),
                    "thinking_process": to_text(live.get("thinking_trace", "")),
                    "raw_output": to_text(live.get("raw_response_full", "")) or response,
                    **{field: live.get(field, "") for field in GENERATION_METADATA_FIELDS},
                }
            )
    return rows



def load_expected_item_ids(benchmark, subset_path):
    path = Path(subset_path or "")
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    task_indices = payload.get("task_indices", {}) if isinstance(payload, dict) else {}
    indices = task_indices.get(benchmark)
    if indices is None:
        key = benchmark_key(benchmark)
        aliases = {
            "mmmu_pro": "mmmu_pro_vision_cot_reasoning",
            "ai2d": "ai2d_reasoning",
            "mathvision_testmini": "mathvision_reason_testmini_reasoning",
        }
        indices = task_indices.get(aliases.get(key, benchmark))
    if indices is None:
        return None
    return {str(int(idx)) for idx in indices}


def validate_rows_against_subset(rows, benchmark, subset_path):
    expected = load_expected_item_ids(benchmark, subset_path or os.getenv("LMMS_FIXED_SUBSET_PATH", ""))
    if expected is None:
        return rows
    filtered = []
    seen = set()
    for row in rows:
        item_id = str(row.get("item_id", "")).strip()
        if item_id not in expected or item_id in seen:
            continue
        seen.add(item_id)
        filtered.append(row)
    missing = sorted(expected - seen, key=lambda x: int(x))
    if missing:
        raise SystemExit(
            f"Answer file is incomplete/corrupt for {benchmark}: "
            f"expected {len(expected)} unique subset rows, got {len(filtered)}; "
            f"missing first IDs: {missing[:10]}"
        )
    return filtered

def evaluate(rows, benchmark, skip_format_failures=False):
    total = len(rows)
    correct = 0
    format_failures = 0
    denominator = 0
    scores = []
    rules = set()
    for row in rows:
        score, failed, rule = score_row(row, benchmark)
        scores.append(score)
        format_failures += 1 if failed else 0
        rules.add(rule)
        if failed and skip_format_failures:
            continue
        denominator += 1
        correct += score
    return {
        "accuracy": correct / denominator if denominator else None,
        "correct": correct,
        "total_rows": total,
        "answered_count": total - format_failures,
        "format_failure_count": format_failures,
        "score_rule": "+".join(sorted(rules)) if rules else "",
        "scores": scores,
    }


def answer_file_path(answers_root, model_name, benchmark, mode, sweep_param, sweep_value, seed):
    model_dir = answers_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    key = benchmark_key(benchmark)
    safe_value = re.sub(r"[^A-Za-z0-9_.-]+", "_", sweep_value)
    safe_seed = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(seed))
    return model_dir / f"{key}__{mode}__{sweep_param}_{safe_value}__seed_{safe_seed}.jsonl"


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
                    "item_id": to_text(record.get("item_id", "")),
                    "seed": to_text(record.get("seed", "")),
                    "true_answer": to_text(record.get("true_answer", "")),
                    "response": to_text(record.get("response", "")),
                    "thinking_process": to_text(record.get("thinking_process", "")),
                    "raw_output": to_text(record.get("raw_output", "")),
                    **{field: record.get(field, "") for field in GENERATION_METADATA_FIELDS},
                }
            )
    return rows


def write_answer_file(rows, answers_root, model_name, benchmark, mode, sweep_param, sweep_value, seed):
    path = answer_file_path(answers_root, model_name, benchmark, mode, sweep_param, sweep_value, seed)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = {
                "item_id": row.get("item_id", ""),
                "seed": row.get("seed", seed),
                "true_answer": row["true_answer"],
                "response": row["response"],
                "raw_output": row.get("raw_output", ""),
            }
            for field in GENERATION_METADATA_FIELDS:
                if field in row:
                    record[field] = row.get(field, "")
            if mode == "reason":
                record["thinking_process"] = row.get("thinking_process", "")
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def result_base(args, metric, row_type):
    key = benchmark_key(args.benchmark)
    return {
        "row_type": row_type,
        "model_name": args.model_name,
        "mode": args.mode,
        "run_name": f"{key}_{args.sweep_param}_{args.sweep_value}",
        "parameters": {
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(float(args.top_k)),
            "repetition_penalty": float(args.repetition_penalty),
            "presence_penalty": float(args.presence_penalty),
        },
        "benchmark": key,
        "accuracy": metric["accuracy"],
        "overall_questions": metric["total_rows"],
        "answered_questions": metric["answered_count"],
        "format_failure_count": metric["format_failure_count"],
        "score_rule": metric["score_rule"],
        "model_id": args.model_id,
        "model_path": args.model_path,
        "max_tokens": int(args.max_tokens) if str(args.max_tokens).isdigit() else args.max_tokens,
        "thinking_budget": int(args.thinking_budget) if str(args.thinking_budget).isdigit() else args.thinking_budget,
        "vllm_version": args.vllm_version,
        "harness_git_commit": args.harness_git_commit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
    }


def self_consistency(rows, benchmark):
    grouped = defaultdict(list)
    for row in rows:
        item = row.get("item_id", "")
        score, failed, _ = score_row(row, benchmark)
        if failed:
            token = f"__format_failure__{row.get('seed', '')}"
        else:
            token = normalize_text(row.get("response", ""))
        grouped[item].append(token)
    vals = []
    for answers in grouped.values():
        n = len(answers)
        if n < 2:
            continue
        agree = 0
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += 1
                if answers[i] and answers[i] == answers[j] and not answers[i].startswith("__format_failure__"):
                    agree += 1
        vals.append(agree / total if total else 0)
    return sum(vals) / len(vals) if vals else None



def vote_token(row, benchmark):
    score, failed, _ = score_row(row, benchmark)
    if failed:
        return ""
    key = benchmark_key(benchmark)
    true_answer = row.get("true_answer", "")
    response = row.get("response", "")
    if key in {"mmmu_pro", "ai2d"} or (key == "mathvision_testmini" and true_answer_is_explicit_letter(true_answer)):
        return extract_final_letter_strict(response)
    return normalize_mathvision_answer(response)


def aggregate_majority_metric(rows, benchmark, k):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("item_id", "")].append(row)

    correct = 0
    answered_items = 0
    format_failures = 0
    rules = set()
    for item_rows in grouped.values():
        tokens = []
        for row in item_rows:
            score, failed, rule = score_row(row, benchmark)
            rules.add(rule)
            if failed:
                format_failures += 1
                continue
            token = vote_token(row, benchmark)
            if token:
                tokens.append(token)

        if not tokens:
            continue

        counts = Counter(tokens)
        token, count = counts.most_common(1)[0]
        total_votes = max(k, len(item_rows))
        if count <= total_votes / 2:
            continue

        answered_items += 1
        true_answer = item_rows[0].get("true_answer", "")
        score, _, _ = score_row({"true_answer": true_answer, "response": token}, benchmark)
        correct += score

    total_items = len(grouped)
    return {
        "accuracy": correct / total_items if total_items else None,
        "correct": correct,
        "total_rows": total_items,
        "answered_count": answered_items,
        "format_failure_count": format_failures,
        "score_rule": "+".join(sorted(rules)) if rules else "",
        "scores": [],
    }

def maybe_append_aggregate(args, answers_root):
    seeds = [s for s in args.sample_seeds.split(",") if s != ""]
    if not seeds:
        return
    all_rows = []
    seed_accuracies = []
    seed_metrics = []
    for seed in seeds:
        path = answer_file_path(answers_root, args.model_name, args.benchmark, args.mode, args.sweep_param, args.sweep_value, seed)
        rows = load_answer_file(path)
        if not rows:
            return
        rows = validate_rows_against_subset(rows, args.benchmark, args.subset_path)
        metric = evaluate(rows, args.benchmark, skip_format_failures=True)
        seed_metrics.append(metric)
        seed_accuracies.append(metric["accuracy"] if metric["accuracy"] is not None else 0.0)
        all_rows.extend(rows)
    total_metric = aggregate_majority_metric(all_rows, args.benchmark, len(seeds))
    record = result_base(args, total_metric, "aggregate")
    record.update(
        {
            "seeds": seeds,
            "k": len(seeds),
            "seed_accuracies": seed_accuracies,
            "accuracy_std": statistics.stdev(seed_accuracies) if len(seed_accuracies) > 1 else 0.0,
            "accuracy_variance": statistics.variance(seed_accuracies) if len(seed_accuracies) > 1 else 0.0,
            "self_consistency": self_consistency(all_rows, args.benchmark),
        }
    )
    append_jsonl(Path(args.results_jsonl), record)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    answers_root = Path(args.answers_root)
    answer_path = answer_file_path(answers_root, args.model_name, args.benchmark, args.mode, args.sweep_param, args.sweep_value, args.seed)
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
        rows = validate_rows_against_subset(rows, args.benchmark, args.subset_path)
    else:
        rows = collect_rows(out_dir, args.benchmark, args.mode, args.seed)
        rows = validate_rows_against_subset(rows, args.benchmark, args.subset_path)
        answer_path = write_answer_file(rows, answers_root, args.model_name, args.benchmark, args.mode, args.sweep_param, args.sweep_value, args.seed)
    metric = evaluate(rows, args.benchmark, skip_format_failures=True)
    record = result_base(args, metric, "seed")
    record["seed"] = args.seed
    append_jsonl(Path(args.results_jsonl), record)
    maybe_append_aggregate(args, answers_root)
    print(f"[INFO] wrote answers={answer_path} accuracy={metric['accuracy']} total={metric['total_rows']} format_failures={metric['format_failure_count']}")


if __name__ == "__main__":
    main()
