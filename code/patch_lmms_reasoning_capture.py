#!/usr/bin/env python3
from pathlib import Path
import importlib.util
import re



def re_replace_method(text, method_name, replacement):
    pattern = rf"    def {re.escape(method_name)}\(.*?\n(?=    def |class |\Z)"
    updated, count = re.subn(pattern, lambda _m: replacement, text, count=1, flags=re.S)
    if count != 1:
        raise SystemExit(f"Could not replace method {method_name}")
    return updated

STRICT_REASONING_PROMPT = (
    "You are a multiple-choice visual reasoning assistant. Think internally if needed. "
    "When your reasoning is complete, output exactly one uppercase option letter from A to J and nothing else. "
    "Do not output words, punctuation, markdown, JSON, or <answer> tags after the reasoning. "
    "The final visible answer must be exactly one letter, for example: B"
)


def patch_task_prompt(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return
    path = Path(spec.origin)
    text = path.read_text(encoding="utf-8")
    start = text.find("SYSTEM_PROMPT = (")
    if start == -1:
        return
    cursor = start + len("SYSTEM_PROMPT = (")
    depth = 1
    while cursor < len(text):
        char = text[cursor]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                cursor += 1
                break
        cursor += 1
    replacement = f'SYSTEM_PROMPT = {STRICT_REASONING_PROMPT!r}\n'
    text = text[:start] + replacement + text[cursor:]

    path.write_text(text, encoding="utf-8")
    print(f"[INFO] Patched strict reasoning prompt: {path}")


for module in (
    "lmms_eval.tasks.ai2d.reasoning.utils",
    "lmms_eval.tasks.mmmu_pro.reasoning.utils",
):
    patch_task_prompt(module)

spec = importlib.util.find_spec("lmms_eval.models.chat.openai")
if spec is None or spec.origin is None:
    raise SystemExit("lmms_eval.models.chat.openai is not importable")

path = Path(spec.origin)
text = path.read_text(encoding="utf-8")
old_response_block = """                    raw_response_text = response.choices[0].message.content
                    response_text = self._normalize_response_output(raw_response_text)
"""
new_response_block = """                    response_message = response.choices[0].message
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
previous_response_block = """                    response_message = response.choices[0].message
                    answer_content = response_message.content or ""
                    reasoning_content = getattr(response_message, "reasoning_content", "") or ""
                    if reasoning_content:
                        raw_response_text = f"<think>{reasoning_content}</think>\\n{answer_content}"
                    else:
                        raw_response_text = answer_content
                    response_text = self._normalize_response_output(answer_content)
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

        gemma_thought = re.search(r"<\|channel>thought\n(.*?)(?:<channel\|>)(.*)$", text, flags=re.I | re.S)
        if gemma_thought:
            return gemma_thought.group(1).strip(), gemma_thought.group(2).strip()

        channel_markers = list(re.finditer(r"<channel\|([^>\n|]+)[^>\n]*>", text, flags=re.I))
        final_markers = [m for m in channel_markers if m.group(1).strip().lower() in {"final", "answer"}]
        if final_markers:
            marker = final_markers[-1]
            thinking = re.sub(r"(?:<\|channel>|<channel\|[^>\n]*>)", " ", text[: marker.start()], flags=re.I).strip()
            final = re.sub(r"(?:<\|channel>|<channel\|[^>\n]*>)", " ", text[marker.end() :], flags=re.I).strip()
            return thinking, final

        stripped = text.strip()
        if os.getenv("LMMS_LIVE_MODE", "") == "reason" and len(stripped) > 500:
            return stripped, ""
        return "", stripped

'''

target_helper = r'''    def _canonical_live_target_text(self, task: str, split: str, doc_id, target_text: str) -> str:
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

'''

helper_anchor = "    def _append_live_answer_row(self, row: dict) -> None:\n"
if "def _split_reasoning_response" not in text:
    if helper_anchor not in text:
        raise SystemExit(f"Could not find helper insertion point in {path}")
    text = text.replace(helper_anchor, split_helper + helper_anchor, 1)
else:
    text = re_replace_method(text, "_split_reasoning_response", split_helper)

if "def _canonical_live_target_text" not in text:
    if helper_anchor not in text:
        raise SystemExit(f"Could not find target helper insertion point in {path}")
    text = text.replace(helper_anchor, target_helper + helper_anchor, 1)
else:
    text = re_replace_method(text, "_canonical_live_target_text", target_helper)

text = text.replace(
    "target_text = self._pick_target_text(task, split, doc_id)\n                    visible_response = self._strip_reasoning_blocks(raw_response_text)",
    "target_text = self._pick_target_text(task, split, doc_id)\n                    target_text = self._canonical_live_target_text(task, split, doc_id, target_text)\n                    visible_response = self._strip_reasoning_blocks(raw_response_text)",
)
text = text.replace(
    'if "mmmu" in task_name or "ai2d" in task_name:\n                        true_letter = self._extract_final_letter(target_text)\n                        pred_letter = self._extract_final_letter(response_text)',
    'if "mmmu" in task_name or "ai2d" in task_name:\n                        true_letter = self._extract_final_letter(target_text)\n                        pred_letter = self._extract_final_letter(response_text)',
)
text = text.replace(
    'if "mmmu" in task_name or "ai2d" in task_name:\n                        match_rule = "letter"',
    'if "mmmu" in task_name or "ai2d" in task_name:\n                        match_rule = "letter"',
)
text = text.replace(
    'if "mmmu" in task_name:\n                        true_letter = self._extract_final_letter(target_text)\n                        pred_letter = self._extract_final_letter(response_text)',
    'if "mmmu" in task_name or "ai2d" in task_name:\n                        true_letter = self._extract_final_letter(target_text)\n                        pred_letter = self._extract_final_letter(response_text)',
)
text = text.replace(
    'if "mmmu" in task_name:\n                        match_rule = "letter"',
    'if "mmmu" in task_name or "ai2d" in task_name:\n                        match_rule = "letter"',
)

if new_response_block in text:
    print(f"[INFO] Reasoning capture already patched: {path}")
elif previous_response_block in text:
    text = text.replace(previous_response_block, new_response_block, 1)
    print(f"[INFO] Upgraded reasoning response splitting: {path}")
elif old_response_block in text:
    text = text.replace(old_response_block, new_response_block, 1)
    print(f"[INFO] Patched reasoning response splitting: {path}")
else:
    raise SystemExit(f"Could not find the expected response extraction block in {path}")

text = text.replace(
    'eb["chat_template_kwargs"] = ctk\n                payload["extra_body"] = eb',
    'eb["chat_template_kwargs"] = ctk\n                if bool(self.enable_thinking):\n                    eb["skip_special_tokens"] = False\n                    eb["spaces_between_special_tokens"] = False\n                payload["extra_body"] = eb',
)

path.write_text(text, encoding="utf-8")
updated = path.read_text(encoding="utf-8")
if "def _append_live_answer_row" not in updated or "thinking_trace" not in updated:
    raise SystemExit("Installed lmms-eval lacks the required live-answer capture hook")

# Runtime payload patch marker is applied by direct text replacement in installed lmms-eval.
