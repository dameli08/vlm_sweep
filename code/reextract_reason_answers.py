#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from gemini_answer_extractor import extract_final_answer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--gemini-model", default="gemini-3.1-flash-lite")
    return parser.parse_args()


def complete_generation(row):
    thinking = str(row.get("thinking_process", "") or "").strip()
    response = str(row.get("response", "") or "").strip()
    if not response or response in thinking[-max(2000, len(response) + 100):]:
        return thinking or response
    return f"{thinking}\n{response}".strip()


def main():
    args = parse_args()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("GEMINI_API_KEY must be exported before re-extraction")

    path = Path(args.jsonl)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    temp = path.with_suffix(path.suffix + ".reextracting")

    with temp.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            generation = complete_generation(row)
            answer = extract_final_answer(
                api_key,
                args.gemini_model,
                args.benchmark,
                generation,
            )
            updated = {
                "true_answer": row.get("true_answer", ""),
                "thinking_process": generation,
                "response": answer,
            }
            handle.write(json.dumps(updated, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(rows)}] response={answer!r}")

    temp.replace(path)
    print(f"[DONE] Re-extracted {len(rows)} rows: {path}")


if __name__ == "__main__":
    main()
