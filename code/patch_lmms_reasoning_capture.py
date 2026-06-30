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


target_helper = r"""    def _canonical_live_target_text(self, task: str, split: str, doc_id, target_text: str) -> str:
        task_name = (task or "").lower()
        if "ai2d" not in task_name:
            return target_text
        try:
            doc = self.task_dict[task][split][doc_id]
        except Exception:
            return target_text
        if not isinstance(doc, dict):
            return target_text
        options = doc.get("options")
        if not isinstance(options, (list, tuple)) or not options:
            return target_text
        answer = doc.get("answer", target_text)
        idx = None
        try:
            idx = int(answer)
        except Exception:
            answer_text = str(answer).strip().upper()
            if re.fullmatch(r"[A-Z]", answer_text):
                idx = ord(answer_text) - ord("A")
        if idx is None:
            try:
                idx = int(target_text)
            except Exception:
                target_letter = self._extract_final_letter(target_text)
                if target_letter:
                    idx = ord(target_letter) - ord("A")
        if idx is not None and 0 <= idx < len(options):
            return f"{chr(ord('A') + idx)}. {options[idx]}"
        return target_text

"""

helper_anchor = "    def _append_live_answer_row(self, row: dict) -> None:\n"
if "def _split_reasoning_response" not in text:
    if helper_anchor not in text:
        raise SystemExit(f"Could not find helper insertion point in {path}")
    text = text.replace(helper_anchor, split_helper + helper_anchor, 1)

if "def _canonical_live_target_text" not in text:
    if helper_anchor not in text:
        raise SystemExit(f"Could not find target helper insertion point in {path}")
    text = text.replace(helper_anchor, target_helper + helper_anchor, 1)

text = text.replace(
    "target_text = self._pick_target_text(task, split, doc_id)\n                    visible_response = self._strip_reasoning_blocks(raw_response_text)",
    "target_text = self._pick_target_text(task, split, doc_id)\n                    target_text = self._canonical_live_target_text(task, split, doc_id, target_text)\n                    visible_response = self._strip_reasoning_blocks(raw_response_text)",
)
text = text.replace(
    'if "mmmu" in task_name:\n                        true_letter = self._extract_final_letter(target_text)\n                        pred_letter = self._extract_final_letter(response_text)',
    'if "mmmu" in task_name or "mmstar" in task_name or "ai2d" in task_name:\n                        true_letter = self._extract_final_letter(target_text)\n                        pred_letter = self._extract_final_letter(response_text)',
)
text = text.replace(
    'if "mmmu" in task_name:\n                        match_rule = "letter"',
    'if "mmmu" in task_name or "mmstar" in task_name or "ai2d" in task_name:\n                        match_rule = "letter"',
)

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
