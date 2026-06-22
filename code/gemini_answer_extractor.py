#!/usr/bin/env python3
import ast
import json
import re
import time
import urllib.error
import urllib.request


def benchmark_key(name):
    lower = (name or "").lower()
    if "mmmu" in lower:
        return "mmmu_pro"
    if "mmstar" in lower:
        return "mmstar"
    if "ocrbench" in lower:
        return "ocrbench"
    return lower


def _response_text(data):
    parts = []
    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            parts.append(part.get("text", ""))
    return "".join(parts)



def _parse_final_answer(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)

    candidates = [text]
    object_match = re.search(r"\{.*?\}", text, flags=re.S)
    if object_match and object_match.group(0) != text:
        candidates.append(object_match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return str(parsed.get("final_answer", "")).strip()
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return str(parsed.get("final_answer", "")).strip()
        except (ValueError, SyntaxError):
            pass

    match = re.search(
        r"[\"']?final_answer[\"']?\s*:\s*[\"']([^\"']*)[\"']",
        text,
        flags=re.I | re.S,
    )
    if match:
        return match.group(1).strip()
    raise ValueError(f"Gemini returned an unreadable extraction: {text[:200]}")

def extract_final_answer(api_key, model, benchmark, model_output, retries=3):
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required for reason-mode answer extraction")

    key = benchmark_key(benchmark)
    if key in {"mmmu_pro", "mmstar"}:
        instruction = (
            "Extract the model's intended final multiple-choice answer. "
            "Return one uppercase letter A-J only. Do not solve the problem yourself and do not "
            "infer an answer from option discussion. Require an explicit final selection, answer tag, "
            "or clear concluding answer. If none exists, return an empty string."
        )
    else:
        instruction = (
            "Extract only the model's intended final OCR answer. Preserve the answer text, "
            "but remove phrases such as 'the answer is', reasoning, quotes, and math delimiters. "
            "Do not solve the problem yourself or infer an answer from unfinished reasoning. "
            "Require an explicit final answer; otherwise return an empty string."
        )

    # The final answer is expected near the end. Bounding this avoids resending very long traces.
    output_tail = (model_output or "")[-16000:]
    prompt = (
        f"{instruction}\n"
        "Return only JSON with this schema: {\"final_answer\": \"...\"}.\n\n"
        f"Model output:\n{output_tail}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {"final_answer": {"type": "STRING"}},
                "required": ["final_answer"],
            },
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    last_error = None
    for attempt in range(retries):
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as result:
                data = json.loads(result.read().decode("utf-8"))
            text = _response_text(data)
            answer = _parse_final_answer(text)
            if key in {"mmmu_pro", "mmstar"}:
                letter = re.search(r"\b([A-J])\b", answer.upper())
                return letter.group(1) if letter else ""
            return answer
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)

    print(f"[WARN] Gemini final-answer extraction failed: {last_error}")
    return ""
