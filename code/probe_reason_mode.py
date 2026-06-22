#!/usr/bin/env python3
"""
Standalone reasoning-mode probe.
Runs independently from run_vlm_benchmark_sweeps.sh.
Connects to an already-running vLLM server and probes MMMU-Pro-Vision.

Behavior:
- Writes CSV rows live after each answered question.
- Stores full, untruncated model response (including think blocks).
- Uses full MMMU-Pro by default (no limit).

Usage:
    conda activate vlm_sweep_20260617
    python probe_reason_mode.py --model qwen3.5-4b
    python probe_reason_mode.py --model qwen3.5-4b --limit 100
"""
import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config (mirrors config.sh defaults; override via CLI args)
# ---------------------------------------------------------------------------
BASE_URL = "http://127.0.0.1:23333/v1"
API_KEY = "EMPTY"
DEFAULT_MODEL = "qwen3.5-4b"
MAX_TOKENS = 12000          # reason budget
TEMPERATURE = 0.2
TOP_P = 0.95
LIMIT = 0                   # 0 means full split

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model",  default=DEFAULT_MODEL)
parser.add_argument("--limit",  type=int, default=LIMIT)
parser.add_argument("--base-url", default=BASE_URL)
parser.add_argument("--out",    default=None,
                    help="Output CSV path (default: probe_reason_<timestamp>.csv)")
parser.add_argument("--allow-fallback", action="store_true",
                    help="Allow fallback text MCQ if MMMU dataset cannot be loaded")
args = parser.parse_args()

out_csv = Path(args.out) if args.out else Path(
    f"probe_reason_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

# ---------------------------------------------------------------------------
# Import openai
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
except ImportError:
    sys.exit("[ERROR] openai package not found. Make sure vlm_sweep_20260617 env is active.")

client = OpenAI(api_key=API_KEY, base_url=args.base_url)

# ---------------------------------------------------------------------------
# Verify server is reachable
# ---------------------------------------------------------------------------
print(f"[probe] Connecting to {args.base_url} ...")
try:
    models = client.models.list()
    available = [m.id for m in models.data]
    print(f"[probe] Available models: {available}")
    if args.model not in available:
        print(f"[WARN] '{args.model}' not in listed models. Will try anyway.")
except Exception as e:
    sys.exit(f"[ERROR] Cannot reach server: {e}")

# ---------------------------------------------------------------------------
# Load a small sample from MMMU-Pro-Vision (HuggingFace datasets)
# ---------------------------------------------------------------------------
print("[probe] Loading MMMU-Pro-Vision samples (standard split) ...")
try:
    from datasets import load_dataset
    config_candidates = [
        "vision",
        "standard (10 options)",
        "standard (4 options)",
        "standard",
    ]
    ds = None
    selected_config = None
    last_err = None
    for cfg in config_candidates:
        try:
            ds = load_dataset("MMMU/MMMU_Pro", cfg, split="test", streaming=True)
            selected_config = cfg
            break
        except Exception as ex:
            last_err = ex

    if ds is None:
        raise RuntimeError(f"No supported MMMU config could be loaded. Last error: {last_err}")

    print(f"[probe] Loaded MMMU config: {selected_config}")
    samples = []
    for item in ds:
        if item.get("answer") and item.get("question"):
            samples.append(item)
        if args.limit > 0 and len(samples) >= args.limit:
            break
except Exception as e:
    if args.allow_fallback:
        print(f"[WARN] Could not load from HuggingFace ({e}). Using built-in fallback questions.")
        samples = None
    else:
        sys.exit(f"[ERROR] Could not load MMMU/MMMU_Pro. Start with internet/HF access or rerun with --allow-fallback. Details: {e}")

# ---------------------------------------------------------------------------
# Fallback: pure text MCQ questions (no image needed) as sanity check
# ---------------------------------------------------------------------------
FALLBACK_QUESTIONS = [
    {
        "question": "What is the primary color produced by mixing red and blue light?",
        "options": {"A": "Yellow", "B": "Magenta", "C": "Cyan", "D": "Green"},
        "answer": "B",
    },
    {
        "question": "Which planet is known as the Red Planet?",
        "options": {"A": "Venus", "B": "Jupiter", "C": "Mars", "D": "Saturn"},
        "answer": "C",
    },
    {
        "question": "What is the chemical symbol for water?",
        "options": {"A": "CO2", "B": "NaCl", "C": "H2O", "D": "O2"},
        "answer": "C",
    },
    {
        "question": "How many sides does a hexagon have?",
        "options": {"A": "5", "B": "6", "C": "7", "D": "8"},
        "answer": "B",
    },
    {
        "question": "Who wrote the play Romeo and Juliet?",
        "options": {"A": "Charles Dickens", "B": "Mark Twain",
                    "C": "William Shakespeare", "D": "Jane Austen"},
        "answer": "C",
    },
]

# ---------------------------------------------------------------------------
# Letter extractor (strict – same logic as backend)
# ---------------------------------------------------------------------------
def extract_letter_strict(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # strip think/analysis blocks first
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
    cleaned = re.sub(r"<analysis>.*?</analysis>", "", cleaned, flags=re.I | re.S)
    upper = cleaned.upper().strip()

    patterns = [
        r"FINAL\s*ANSWER\s*(?:IS|:|=|-)\s*\(?\s*([A-Z])\s*\)?",
        r"ANSWER\s*(?:IS|:|=|-)\s*\(?\s*([A-Z])\s*\)?",
        r"OPTION\s*(?:IS|:|=|-)\s*\(?\s*([A-Z])\s*\)?",
        r"CHOICE\s*(?:IS|:|=|-)\s*\(?\s*([A-Z])\s*\)?",
        r"CORRECT\s*CHOICE\s*(?:IS|:|=|-)\s*\(?\s*([A-Z])\s*\)?",
    ]
    for pat in patterns:
        hits = re.findall(pat, upper)
        if hits:
            return hits[-1]

    solo = re.fullmatch(r"\(?\s*([A-Z])\s*[\)\].,:;\-!?]*\s*", upper)
    if solo:
        return solo.group(1)

    return ""

def has_think_block(text: str) -> bool:
    return bool(re.search(r"<think>", text, re.I))

def think_block_length(text: str) -> int:
    m = re.search(r"<think>(.*?)</think>", text, re.I | re.S)
    return len(m.group(1)) if m else 0

# ---------------------------------------------------------------------------
# Build a question into messages
# ---------------------------------------------------------------------------
def build_messages(q: dict, is_fallback: bool) -> list:
    if is_fallback:
        opts = "\n".join(f"({k}) {v}" for k, v in q["options"].items())
        user_text = (
            f"{q['question']}\n\n{opts}\n\n"
            "Think carefully, then output your final answer as a single uppercase letter "
            "in the format: Answer: X"
        )
        return [{"role": "user", "content": user_text}]

    # HuggingFace MMMU-Pro sample
    opts_raw = q.get("options", "")
    if isinstance(opts_raw, str):
        try:
            opts_raw = json.loads(opts_raw)
        except Exception:
            pass
    if isinstance(opts_raw, list):
        labels = "ABCDEFGHIJ"
        opts_str = "\n".join(f"({labels[i]}) {v}" for i, v in enumerate(opts_raw))
    elif isinstance(opts_raw, dict):
        opts_str = "\n".join(f"({k}) {v}" for k, v in opts_raw.items())
    else:
        opts_str = str(opts_raw)

    content_parts = []

    # attach image if present
    img = q.get("image") or q.get("image_1")
    if img is not None:
        try:
            import base64
            from io import BytesIO
            buf = BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        except Exception as ex:
            print(f"  [WARN] Could not encode image: {ex}")

    content_parts.append({
        "type": "text",
        "text": (
            f"{q['question']}\n\n{opts_str}\n\n"
            "Think carefully, then output your final answer as a single uppercase letter "
            "in the format: Answer: X"
        ),
    })
    return [{"role": "user", "content": content_parts}]

# ---------------------------------------------------------------------------
# Probe loop
# ---------------------------------------------------------------------------
use_fallback = samples is None
if use_fallback:
    question_list = FALLBACK_QUESTIONS[: args.limit] if args.limit > 0 else FALLBACK_QUESTIONS
else:
    question_list = samples
source_label = "fallback_text_mcq" if use_fallback else "mmmu_pro_vision"

print(f"\n[probe] Source: {source_label}  |  Model: {args.model}  |  "
      f"Questions: {len(question_list)}  |  max_tokens={MAX_TOKENS}\n")
print("=" * 80)

rows = []
correct = 0
empty = 0

fieldnames = [
    "idx",
    "source",
    "model",
    "true_answer",
    "raw_response",
    "has_think_block",
    "think_block_chars",
    "visible_output",
    "predicted_letter",
    "match",
    "elapsed_s",
    "error",
]

with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()

for idx, q in enumerate(question_list):
    true_answer = str(q.get("answer", "")).strip().upper()
    messages = build_messages(q, use_fallback)

    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": True}
        },
    }

    print(f"[{idx+1}/{len(question_list)}] Sending ... (true answer: {true_answer})")
    t0 = time.time()
    raw_response = ""
    error = ""
    try:
        resp = client.chat.completions.create(**payload)
        raw_response = resp.choices[0].message.content or ""
        elapsed = round(time.time() - t0, 1)
    except Exception as ex:
        error = str(ex)
        elapsed = round(time.time() - t0, 1)
        print(f"  [ERROR] {error}")

    predicted_letter = extract_letter_strict(raw_response)
    has_think = has_think_block(raw_response)
    think_len = think_block_length(raw_response)
    match = (predicted_letter == true_answer) if (predicted_letter and true_answer) else None

    if not predicted_letter:
        empty += 1
    elif match:
        correct += 1

    # pretty print
    think_excerpt = ""
    if has_think:
        m = re.search(r"<think>(.*?)</think>", raw_response, re.I | re.S)
        if m:
            think_excerpt = m.group(1).strip()[:200].replace("\n", " ")
    visible = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.I | re.S).strip()

    print(f"  has_think_block : {has_think}  (think length: {think_len} chars)")
    if think_excerpt:
        print(f"  think (first 200): {think_excerpt}")
    print(f"  visible output  : {visible[:200]}")
    print(f"  predicted_letter: '{predicted_letter}'  |  true: '{true_answer}'  |  "
          f"match: {match}  |  elapsed: {elapsed}s")
    if error:
        print(f"  error: {error}")
    print()

    row = {
        "idx": idx + 1,
        "source": source_label,
        "model": args.model,
        "true_answer": true_answer,
        "raw_response": raw_response,
        "has_think_block": has_think,
        "think_block_chars": think_len,
        "visible_output": visible,
        "predicted_letter": predicted_letter,
        "match": match,
        "elapsed_s": elapsed,
        "error": error,
    }
    rows.append(row)

    # Live CSV append: write each answered question immediately.
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(row)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total = len(rows)
answered = total - empty
print("=" * 80)
print(f"SUMMARY  total={total}  answered={answered}  correct={correct}  empty={empty}")
print(f"Accuracy (answered only): {correct}/{answered} = "
      f"{correct/answered*100:.1f}%" if answered else "N/A")
print()

print(f"[probe] Results saved to: {out_csv.resolve()}")
