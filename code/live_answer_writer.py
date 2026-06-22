#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from gemini_answer_extractor import extract_final_answer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--benchmark", default="")
    parser.add_argument("--gemini-model", default="gemini-3.1-flash-lite")
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_rows(path):
    if not path.exists():
        return []
    try:
        csv.field_size_limit(sys.maxsize)
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def output_row(row, mode, benchmark, gemini_api_key, gemini_model):
    raw_response = (
        row.get("raw_response_full")
        or row.get("visible_response")
        or row.get("prediction_clean")
        or row.get("prediction_raw", "")
    )
    if mode == "reason":
        final_answer = extract_final_answer(
            gemini_api_key,
            gemini_model,
            benchmark,
            raw_response,
        )
        return {
            "true_answer": row.get("true_answer_text", ""),
            "thinking_process": raw_response.strip(),
            "response": final_answer,
        }

    return {
        "true_answer": row.get("true_answer_text", ""),
        "response": (row.get("visible_response") or row.get("prediction_clean") or row.get("prediction_raw", "")).strip(),
    }


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
    ready_path = stop_path.with_name(".gemini_answers_ready")
    ready_path.unlink(missing_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        submitted = existing_count(jsonl_path)
    else:
        jsonl_path.write_text("", encoding="utf-8")
        submitted = 0

    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if args.mode == "reason" and not gemini_api_key:
        raise SystemExit("GEMINI_API_KEY is required for reason-mode answer extraction")

    max_workers = max(1, args.workers)
    max_in_flight = max_workers * 3
    futures = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            rows = read_rows(csv_path)
            while submitted < len(rows) and len(futures) < max_in_flight:
                row = rows[submitted]
                benchmark = args.benchmark or row.get("benchmark", "")
                future = executor.submit(
                    output_row,
                    row,
                    args.mode,
                    benchmark,
                    gemini_api_key,
                    args.gemini_model,
                )
                futures[future] = submitted
                submitted += 1

            if futures:
                done, _ = wait(futures, timeout=0.2, return_when=FIRST_COMPLETED)
                for future in done:
                    futures.pop(future, None)
                    append_record(jsonl_path, future.result())
            else:
                time.sleep(0.1)

            if stop_path.exists():
                rows = read_rows(csv_path)
                if submitted >= len(rows) and not futures:
                    if args.mode == "reason":
                        ready_path.touch()
                    break


if __name__ == "__main__":
    main()
