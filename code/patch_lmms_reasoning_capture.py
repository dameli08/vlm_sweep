#!/usr/bin/env python3
from pathlib import Path
import importlib.util

spec = importlib.util.find_spec("lmms_eval.models.chat.openai")
if spec is None or spec.origin is None:
    raise SystemExit("lmms_eval.models.chat.openai is not importable")

path = Path(spec.origin)
text = path.read_text(encoding="utf-8")
old = """                    raw_response_text = response.choices[0].message.content
                    response_text = self._normalize_response_output(raw_response_text)
"""
new = """                    response_message = response.choices[0].message
                    answer_content = response_message.content or ""
                    explicit_reasoning = getattr(response_message, "reasoning_content", "") or ""
                    thinking_content, final_content = self._split_reasoning_response(
                        answer_content, explicit_reasoning
                    )
                    if thinking_content:
                        raw_response_text = f"<think>{thinking_content}</think>\\n{final_content}"
                    else:
                        raw_response_text = final_content
                    response_text = self._normalize_response_output(final_content)
"""

split_helper = r'''    def _split_reasoning_response(self, text: str, explicit_reasoning: str = ""):
        text = text if isinstance(text, str) else ""
        explicit_reasoning = explicit_reasoning if isinstance(explicit_reasoning, str) else ""
        if explicit_reasoning.strip():
            return explicit_reasoning.strip(), self._strip_reasoning_blocks(text).strip()

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
        stripped = text.strip()
        if os.getenv("LMMS_LIVE_MODE", "") == "reason" and len(stripped) > 500:
            return stripped, ""
        return "", stripped

'''

helper_anchor = "    def _append_live_answer_row(self, row: dict) -> None:\n"
if "def _split_reasoning_response" not in text:
    if helper_anchor not in text:
        raise SystemExit(f"Could not find helper insertion point in {path}")
    text = text.replace(helper_anchor, split_helper + helper_anchor, 1)

previous = """                    response_message = response.choices[0].message
                    answer_content = response_message.content or ""
                    reasoning_content = getattr(response_message, "reasoning_content", "") or ""
                    if reasoning_content:
                        raw_response_text = f"<think>{reasoning_content}</think>\\n{answer_content}"
                    else:
                        raw_response_text = answer_content
                    response_text = self._normalize_response_output(answer_content)
"""

if new in text:
    print(f"[INFO] Reasoning capture already patched: {path}")
elif previous in text:
    text = text.replace(previous, new, 1)
    print(f"[INFO] Upgraded reasoning response splitting: {path}")
elif old in text:
    text = text.replace(old, new, 1)
    print(f"[INFO] Patched reasoning response splitting: {path}")
else:
    raise SystemExit(f"Could not find the expected response extraction block in {path}")

path.write_text(text, encoding="utf-8")
updated = path.read_text(encoding="utf-8")
if "def _append_live_answer_row" not in updated or "thinking_trace" not in updated:
    raise SystemExit("Installed lmms-eval lacks the required live-answer capture hook")
