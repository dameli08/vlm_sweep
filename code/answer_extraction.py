#!/usr/bin/env python3
import re


def to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _strip_channel_tags(text):
    return re.sub(r"(?:<\|channel>|<channel\|[^>\n]*>)", " ", text, flags=re.I)


def split_reasoning_response(text):
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

    gemma_thought = re.search(r"<\|channel>thought\n(.*?)(?:<channel\|>)(.*)$", text, flags=re.I | re.S)
    if gemma_thought:
        return gemma_thought.group(1).strip(), gemma_thought.group(2).strip()

    channel_markers = list(re.finditer(r"<channel\|([^>\n|]+)[^>\n]*>", text, flags=re.I))
    final_markers = [m for m in channel_markers if m.group(1).strip().lower() in {"final", "answer"}]
    if final_markers:
        marker = final_markers[-1]
        thinking = _strip_channel_tags(text[: marker.start()]).strip()
        final = _strip_channel_tags(text[marker.end() :]).strip()
        return thinking, final

    return "", text.strip()


def strip_reasoning_blocks(text):
    return split_reasoning_response(text)[1]


def extract_final_letter_strict(text):
    raw = strip_reasoning_blocks(to_text(text)).strip()
    raw = re.sub(r"</?(think|analysis|answer)>", " ", raw, flags=re.I)
    raw = _strip_channel_tags(raw).strip()
    patterns = [
        r"^\s*\(?\s*([A-J])\s*\)?\s*[\.)\]:,;!-]*\s*$",
        r"<answer>\s*\(?\s*([A-J])\s*\)?\s*</answer>",
        r"(?:final\s+answer|answer|option|choice|final\s+selection)\s*(?:is|:|=|-)?\s*\(?\s*([A-J])\s*\)?\s*(?:$|[\.)\],:;!])",
        r"^\s*([A-J])\s*[\.)\]:-]\s+",
    ]
    for pattern in patterns:
        hits = re.findall(pattern, raw, flags=re.I | re.S)
        if hits:
            return hits[-1].upper()

    # Last-line fallback, still strict: a single letter alone at the end.
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if lines:
        last = lines[-1]
        solo = re.fullmatch(r"\(?\s*([A-J])\s*\)?\s*[\.)\]:,;!-]*", last, flags=re.I)
        if solo:
            return solo.group(1).upper()
    return ""


def extract_reasoning_answer(text):
    thinking, final = split_reasoning_response(text)
    answer = extract_final_letter_strict(final)
    if not answer:
        answer = extract_final_letter_strict(text)
    return thinking, answer
