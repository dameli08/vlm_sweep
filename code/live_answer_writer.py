#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from answer_extraction import extract_reasoning_answer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--benchmark", default="")
    parser.add_argument("--stop-file", required=True)
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


def output_row(row, mode):
    raw_response = (
        row.get("raw_response_full")
        or row.get("visible_response")
        or row.get("prediction_clean")
        or row.get("prediction_raw", "")
    )
    if mode == "reason":
        thinking, final_answer = extract_reasoning_answer(raw_response)
        return {
            "true_answer": row.get("true_answer_text", ""),
            "thinking_process": thinking.strip(),
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
    ready_path = stop_path.with_name(".answers_ready")
    ready_path.unlink(missing_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        submitted = existing_count(jsonl_path)
    else:
        jsonl_path.write_text("", encoding="utf-8")
        submitted = 0

    while True:
        rows = read_rows(csv_path)
        while submitted < len(rows):
            append_record(jsonl_path, output_row(rows[submitted], args.mode))
            submitted += 1

        if stop_path.exists():
            rows = read_rows(csv_path)
            while submitted < len(rows):
                append_record(jsonl_path, output_row(rows[submitted], args.mode))
                submitted += 1
            if args.mode == "reason":
                ready_path.touch()
            break

        time.sleep(0.1)


if __name__ == "__main__":
    main()
