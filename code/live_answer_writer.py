#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from answer_extraction import extract_benchmark_answer, extract_reasoning_benchmark_answer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--benchmark", default="")
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_rows(path, expected_benchmark="", expected_mode=""):
    if not path.exists():
        return []
    try:
        csv.field_size_limit(sys.maxsize)
        with path.open(encoding="utf-8", newline="") as handle:
            rows = []
            for row in csv.DictReader(handle):
                if not isinstance(row, dict) or None in row:
                    continue
                if expected_benchmark and row.get("benchmark") != expected_benchmark:
                    continue
                if expected_mode and row.get("mode") != expected_mode:
                    continue
                doc_id = str(row.get("doc_id", "")).strip()
                if not doc_id.isdigit():
                    continue
                if not row.get("true_answer_text"):
                    continue
                rows.append(row)
            return rows
    except (OSError, csv.Error):
        return []


def token_count(text):
    return len(str(text or "").split())


def loop_score(text, n=5, min_repeats=3, max_gap=2):
    tokens = str(text or "").split()
    if len(tokens) < n:
        return 0.0
    positions = {}
    for idx in range(len(tokens) - n + 1):
        gram = tuple(tokens[idx : idx + n])
        positions.setdefault(gram, []).append(idx)
    covered = set()
    for idxs in positions.values():
        if len(idxs) < min_repeats:
            continue
        run = [idxs[0]]
        for idx in idxs[1:]:
            if idx - run[-1] <= n + max_gap:
                run.append(idx)
            else:
                if len(run) >= min_repeats:
                    for start in run:
                        covered.update(range(start, min(start + n, len(tokens))))
                run = [idx]
        if len(run) >= min_repeats:
            for start in run:
                covered.update(range(start, min(start + n, len(tokens))))
    return len(covered) / len(tokens) if tokens else 0.0


def base_metadata(row, thinking, response, raw_response):
    finish_reason = row.get("finish_reason", "")
    thinking_budget = os.getenv("LMMS_THINKING_BUDGET", "")
    max_tokens = os.getenv("LMMS_MAX_TOKENS", "")
    return {
        "finish_reason": finish_reason,
        "budget_truncation_flag": bool(finish_reason == "length"),
        "trace_token_count": token_count(thinking),
        "answer_token_count": token_count(response),
        "loop_score": loop_score(thinking),
        "max_tokens": int(max_tokens) if str(max_tokens).isdigit() else max_tokens,
        "thinking_budget": int(thinking_budget) if str(thinking_budget).isdigit() else thinking_budget,
        "model_id": os.getenv("LMMS_MODEL_ALIAS", row.get("model_alias", "")),
        "model_path": os.getenv("LMMS_MODEL_PATH", ""),
        "vllm_version": os.getenv("LMMS_VLLM_VERSION", ""),
        "harness_git_commit": os.getenv("LMMS_HARNESS_GIT_COMMIT", ""),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "request_success": row.get("request_success", ""),
        "input_tokens": int(row.get("input_tokens", "0") or 0),
        "output_tokens": int(row.get("output_tokens", "0") or 0),
        "reasoning_tokens": int(row.get("reasoning_tokens", "0") or 0),
        "first_64_token_logprobs": row.get("first_64_token_logprobs", ""),
    }


def output_row(row, mode):
    raw_response = (
        row.get("raw_response_full")
        or row.get("visible_response")
        or row.get("prediction_clean")
        or row.get("prediction_raw", "")
    )
    true_answer = row.get("true_answer_text", "")
    benchmark = row.get("benchmark", "")
    if mode == "reason":
        thinking, final_answer = extract_reasoning_benchmark_answer(raw_response, benchmark, true_answer)
        record = {
            "item_id": row.get("doc_id", ""),
            "seed": os.getenv("LMMS_SAMPLE_SEED", ""),
            "true_answer": true_answer,
            "thinking_process": thinking.strip(),
            "raw_output": raw_response,
            "response": final_answer,
        }
        record.update(base_metadata(row, thinking, final_answer, raw_response))
        return record

    visible = row.get("visible_response") or row.get("prediction_clean") or row.get("prediction_raw", "")
    final_answer = extract_benchmark_answer(visible, benchmark, true_answer)
    record = {
        "item_id": row.get("doc_id", ""),
        "seed": os.getenv("LMMS_SAMPLE_SEED", ""),
        "true_answer": true_answer,
        "raw_output": visible,
        "response": final_answer,
    }
    record.update(base_metadata(row, "", final_answer, visible))
    return record


def append_record(path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def existing_count(path):
    if not path.exists():
        return 0
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    jsonl_path = Path(args.jsonl)
    stop_path = Path(args.stop_file)
    ready_path = stop_path.with_name(".answers_ready")
    ready_path.unlink(missing_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        submitted = existing_count(jsonl_path)
    else:
        jsonl_path.write_text("", encoding="utf-8")
        submitted = 0

    while True:
        rows = read_rows(csv_path, args.benchmark, args.mode)
        while submitted < len(rows):
            append_record(jsonl_path, output_row(rows[submitted], args.mode))
            submitted += 1

        if stop_path.exists():
            rows = read_rows(csv_path, args.benchmark, args.mode)
            while submitted < len(rows):
                append_record(jsonl_path, output_row(rows[submitted], args.mode))
                submitted += 1
            if args.mode == "reason":
                ready_path.touch()
            break

        time.sleep(0.1)


if __name__ == "__main__":
    main()
