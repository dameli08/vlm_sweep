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

STRICT_MCQ_REASONING_PROMPT = (
    "You are a multiple-choice visual reasoning assistant. Think internally if needed. "
    "When your reasoning is complete, output exactly one uppercase option letter from A to J and nothing else. "
    "Do not output words, punctuation, markdown, JSON, or <answer> tags after the reasoning. "
    "The final visible answer must be exactly one letter, for example: B"
)

STRICT_MATHVISION_REASONING_PROMPT = (
    "You are a visual math reasoning assistant. Think internally if needed. "
    "When reasoning is complete, output only the final answer and nothing else. "
    "If the problem has choices, output exactly one uppercase option letter from A to J. "
    "If the problem is open-ended, output exactly one concise mathematical value or expression, preferably inside one \\boxed{} expression. "
    "Do not output explanatory words, markdown, JSON, or <answer> tags after reasoning."
)


def patch_task_prompt(module_name, prompt):
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
    replacement = f'SYSTEM_PROMPT = {prompt!r}\n'
    text = text[:start] + replacement + text[cursor:]
    path.write_text(text, encoding="utf-8")
    print(f"[INFO] Patched strict reasoning prompt: {path}")


for module in (
    "lmms_eval.tasks.ai2d.reasoning.utils",
    "lmms_eval.tasks.mmmu_pro.reasoning.utils",
):
    patch_task_prompt(module, STRICT_MCQ_REASONING_PROMPT)
patch_task_prompt("lmms_eval.tasks.mathvision.reasoning.utils", STRICT_MATHVISION_REASONING_PROMPT)

# Tighten MathVision prompts without changing the underlying dataset/task.
def patch_mathvision_doc_prompt(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return
    path = Path(spec.origin)
    text = path.read_text(encoding="utf-8")
    replacement = r'''def mathvision_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc.get("question", "")
    choices = doc.get("options") or []
    if not isinstance(choices, (list, tuple)):
        choices = []
    options = [chr(ord("A") + i) for i in range(len(choices))]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])

    if choices_str:
        return (
            f"{question}\nChoices:\n{choices_str}\n"
            "Think carefully. After your reasoning, output exactly one uppercase option letter from A to J and nothing else."
        )
    return (
        f"{question}\n"
        "Think carefully. After your reasoning, output only the final mathematical answer. "
        "Use one \\boxed{} expression if possible, and do not add explanatory words after it."
    )

'''
    text, count = re.subn(r"def mathvision_doc_to_text\(doc, lmms_eval_specific_kwargs=None\):\n.*?\n\n(?=def mathvision_|mathvision_doc_to_messages)", lambda _m: replacement, text, count=1, flags=re.S)
    if count:
        path.write_text(text, encoding="utf-8")
        print(f"[INFO] Patched MathVision doc prompt: {path}")


patch_mathvision_doc_prompt("lmms_eval.tasks.mathvision.utils")
patch_mathvision_doc_prompt("lmms_eval.tasks.mathvision.reasoning.utils")


def patch_mathvision_reasoning_template():
    spec = importlib.util.find_spec("lmms_eval.tasks.mathvision.reasoning.utils")
    if spec is None or spec.origin is None:
        return
    task_dir = Path(spec.origin).parent
    template_path = task_dir / "_default_template_yaml"
    template = 'output_type: generate_until\ndoc_to_visual: !function utils.mathvision_doc_to_visual\ndoc_to_text: !function utils.mathvision_doc_to_text\ndoc_to_messages: !function utils.mathvision_doc_to_messages\ndoc_to_target: "answer"\nprocess_results: !function utils.mathvision_process_results\nmetric_list:\n  - metric: acc_score\n    aggregation: mean\n    higher_is_better: true\n  - metric: format_score\n    aggregation: mean\n    higher_is_better: true\ngeneration_kwargs:\n  max_new_tokens: 32768\nmetadata:\n  version: 0.0\n'
    if not template_path.exists() or template_path.read_text(encoding="utf-8") != template:
        template_path.write_text(template, encoding="utf-8")
        print(f"[INFO] Patched MathVision reasoning template: {template_path}")


patch_mathvision_reasoning_template()

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
        if "ai2d" not in task_name and "mathvision" not in task_name:
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
    'if "mmmu" in task_name or "ai2d" in task_name or ("mathvision" in task_name and self._extract_final_letter(target_text)):\n                        true_letter = self._extract_final_letter(target_text)\n                        pred_letter = self._extract_final_letter(response_text)',
)
text = text.replace(
    'if "mmmu" in task_name or "ai2d" in task_name:\n                        match_rule = "letter"',
    'if "mmmu" in task_name or "ai2d" in task_name or ("mathvision" in task_name and self._extract_final_letter(target_text)):\n                        match_rule = "letter"',
)
text = text.replace(
    'if "mmmu" in task_name:\n                        true_letter = self._extract_final_letter(target_text)\n                        pred_letter = self._extract_final_letter(response_text)',
    'if "mmmu" in task_name or "ai2d" in task_name or ("mathvision" in task_name and self._extract_final_letter(target_text)):\n                        true_letter = self._extract_final_letter(target_text)\n                        pred_letter = self._extract_final_letter(response_text)',
)
text = text.replace(
    'if "mmmu" in task_name:\n                        match_rule = "letter"',
    'if "mmmu" in task_name or "ai2d" in task_name or ("mathvision" in task_name and self._extract_final_letter(target_text)):\n                        match_rule = "letter"',
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

if 'if "presence_penalty" in request_gen_kwargs:' not in text:
    text = text.replace(
        'if "repetition_penalty" in request_gen_kwargs:\n                eb = payload.get("extra_body", {})\n                if not isinstance(eb, dict):\n                    eb = {}\n                eb["repetition_penalty"] = request_gen_kwargs.get("repetition_penalty")\n                payload["extra_body"] = eb\n\n',
        'if "repetition_penalty" in request_gen_kwargs:\n                eb = payload.get("extra_body", {})\n                if not isinstance(eb, dict):\n                    eb = {}\n                eb["repetition_penalty"] = request_gen_kwargs.get("repetition_penalty")\n                payload["extra_body"] = eb\n\n            if "presence_penalty" in request_gen_kwargs:\n                payload["presence_penalty"] = request_gen_kwargs.get("presence_penalty")\n\n            if "seed" in request_gen_kwargs:\n                payload["seed"] = int(request_gen_kwargs.get("seed"))\n\n',
    )


# Add per-generation metadata to live-answer capture.
text = text.replace(
    'if payload is None:\n                return "", local_index, False, False, 0.0, 0, 0, 0',
    'if payload is None:\n                return "", "", local_index, False, False, 0.0, 0, 0, 0, "", 0, 0',
)
text = text.replace(
    'finish_reason = getattr(response.choices[0], "finish_reason", "") or ""\n                    response_message = response.choices[0].message',
    'response_message = response.choices[0].message',
)
text = text.replace(
    'response_message = response.choices[0].message\n                    answer_content = response_message.content or ""',
    'finish_reason = getattr(response.choices[0], "finish_reason", "") or ""\n                    response_message = response.choices[0].message\n                    answer_content = response_message.content or ""',
    1,
)
text = text.replace(
    """                        reasoning_tokens,\n                    )""",
    """                        reasoning_tokens,\n                        finish_reason,\n                        output_tokens,\n                        input_tokens,\n                    )""",
    1,
)
text = text.replace(
    'return failure_content, failure_content, local_index, False, rate_limited, elapsed, 0, 0, 0',
    'return failure_content, failure_content, local_index, False, rate_limited, elapsed, 0, 0, 0, "error", 0, 0',
)
text = text.replace(
    """                        reasoning_tokens,\n                    ) = future.result()""",
    """                        reasoning_tokens,\n                        finish_reason,\n                        output_tokens,\n                        input_tokens,\n                    ) = future.result()""",
    1,
)
text = text.replace(
    '"resolved_match",\n        ]',
    '"resolved_match",\n            "finish_reason",\n            "request_success",\n            "input_tokens",\n            "output_tokens",\n            "reasoning_tokens",\n        ]',
)
text = text.replace(
    """                            "resolved_match": resolved_match,\n                        }""",
    """                            "resolved_match": resolved_match,\n                            "finish_reason": finish_reason,\n                            "request_success": "1" if success else "0",\n                            "input_tokens": input_tokens,\n                            "output_tokens": output_tokens,\n                            "reasoning_tokens": reasoning_tokens,\n                        }""",
    1,
)
text = text.replace(
    'writer = csv.DictWriter(f, fieldnames=fieldnames)',
    'writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, escapechar="\\\\")',
)

path.write_text(text, encoding="utf-8")
updated = path.read_text(encoding="utf-8")
if "def _append_live_answer_row" not in updated or "thinking_trace" not in updated:
    raise SystemExit("Installed lmms-eval lacks the required live-answer capture hook")

# Patch fixed-subset filtering into Task.build_all_requests.
task_spec = importlib.util.find_spec("lmms_eval.api.task")
if task_spec is None or task_spec.origin is None:
    raise SystemExit("lmms_eval.api.task is not importable")
task_path = Path(task_spec.origin)
task_text = task_path.read_text(encoding="utf-8")
old_subset_block = '''        doc_id_docs = utils.create_iterator(
            enumerate(self.eval_docs_no_media),
            rank=rank,
            limit=int(limit) if limit else None,
            world_size=world_size,
            offset=offset,
        )
        doc_iterator_for_counting = (
            utils.create_iterator(
                range(len(self.test_docs())),
                rank=rank,
                limit=limit,
                world_size=world_size,
                offset=offset,
            )
            if self.has_test_docs()
            else utils.create_iterator(
                range(len(self.validation_docs())),
                rank=rank,
                limit=limit,
                world_size=world_size,
                offset=offset,
            )
        )

        num_docs = sum(1 for _ in doc_iterator_for_counting)
'''
new_subset_block = '''        fixed_subset_path = os.getenv("LMMS_FIXED_SUBSET_PATH", "").strip()
        fixed_indices = None
        if fixed_subset_path:
            try:
                with open(fixed_subset_path, "r", encoding="utf-8") as subset_f:
                    subset_payload = json.load(subset_f)
                fixed_indices = subset_payload.get("task_indices", {}).get(self.config.task)
            except Exception as exc:
                eval_logger.warning(f"Could not load fixed subset {fixed_subset_path}: {exc}")
                fixed_indices = None

        if fixed_indices is not None:
            selected_indices = [int(i) for i in fixed_indices if int(i) >= offset]
            selected_indices = selected_indices[rank::world_size]
            if limit:
                selected_indices = selected_indices[: int(limit)]
            doc_id_docs = [(idx, self.eval_docs_no_media[idx]) for idx in selected_indices if idx < len(self.eval_docs_no_media)]
            num_docs = len(doc_id_docs)
        else:
            doc_id_docs = utils.create_iterator(
                enumerate(self.eval_docs_no_media),
                rank=rank,
                limit=int(limit) if limit else None,
                world_size=world_size,
                offset=offset,
            )
            doc_iterator_for_counting = (
                utils.create_iterator(
                    range(len(self.test_docs())),
                    rank=rank,
                    limit=limit,
                    world_size=world_size,
                    offset=offset,
                )
                if self.has_test_docs()
                else utils.create_iterator(
                    range(len(self.validation_docs())),
                    rank=rank,
                    limit=limit,
                    world_size=world_size,
                    offset=offset,
                )
            )

            num_docs = sum(1 for _ in doc_iterator_for_counting)
'''
if "LMMS_FIXED_SUBSET_PATH" not in task_text:
    if old_subset_block not in task_text:
        raise SystemExit(f"Could not find subset insertion block in {task_path}")
    task_text = task_text.replace(old_subset_block, new_subset_block, 1)
    task_path.write_text(task_text, encoding="utf-8")
    print(f"[INFO] Patched fixed subset support: {task_path}")
else:
    print(f"[INFO] Fixed subset support already patched: {task_path}")


# Ensure first-64-token logprobs capture stays installed.
def patch_first_64_logprobs_capture():
    openai_spec = importlib.util.find_spec("lmms_eval.models.chat.openai")
    if openai_spec is None or openai_spec.origin is None:
        return
    openai_path = Path(openai_spec.origin)
    openai_text = openai_path.read_text(encoding="utf-8")
    if "import json\n" not in openai_text:
        openai_text = openai_text.replace("import csv\n", "import csv\nimport json\n", 1)

    helper = """
    def _first_token_logprobs_json(self, response, limit: int = 64) -> str:
        try:
            logprobs = getattr(response.choices[0], \"logprobs\", None)
            content = getattr(logprobs, \"content\", None) or []
            packed = []
            for item in content[:limit]:
                token = getattr(item, \"token\", \"\")
                logprob = getattr(item, \"logprob\", None)
                top_items = []
                for top in (getattr(item, \"top_logprobs\", None) or [])[:5]:
                    top_items.append({\"token\": getattr(top, \"token\", \"\"), \"logprob\": getattr(top, \"logprob\", None)})
                packed.append({\"token\": token, \"logprob\": logprob, \"top_logprobs\": top_items})
            return json.dumps(packed, ensure_ascii=False)
        except Exception:
            return \"\"

"""
    if "def _first_token_logprobs_json" not in openai_text:
        openai_text = openai_text.replace(
            "    def _append_live_answer_row(self, row: dict) -> None:\n",
            helper + "    def _append_live_answer_row(self, row: dict) -> None:\n",
            1,
        )
    if '\"first_64_token_logprobs\",' not in openai_text:
        openai_text = openai_text.replace(
            '            "reasoning_tokens",\n        ]\n',
            '            "reasoning_tokens",\n            "first_64_token_logprobs",\n        ]\n',
            1,
        )
    if 'return "", "", local_index, False, False, 0.0, 0, 0, 0, "", 0, 0, ""' not in openai_text:
        openai_text = openai_text.replace(
            'return "", "", local_index, False, False, 0.0, 0, 0, 0, "", 0, 0',
            'return "", "", local_index, False, False, 0.0, 0, 0, 0, "", 0, 0, ""',
            1,
        )
    if 'return failure_content, failure_content, local_index, False, rate_limited, elapsed, 0, 0, 0, "error", 0, 0, ""' not in openai_text:
        openai_text = openai_text.replace(
            'return failure_content, failure_content, local_index, False, rate_limited, elapsed, 0, 0, 0, "error", 0, 0',
            'return failure_content, failure_content, local_index, False, rate_limited, elapsed, 0, 0, 0, "error", 0, 0, ""',
            1,
        )
    if "first_64_token_logprobs = self._first_token_logprobs_json(response, 64)" not in openai_text:
        openai_text = openai_text.replace(
            'finish_reason = getattr(response.choices[0], "finish_reason", "") or ""\n                    response_message = response.choices[0].message',
            'finish_reason = getattr(response.choices[0], "finish_reason", "") or ""\n                    first_64_token_logprobs = self._first_token_logprobs_json(response, 64)\n                    response_message = response.choices[0].message',
            1,
        )
    openai_text = openai_text.replace(
        '                        output_tokens,\n                        input_tokens,\n                    )\n',
        '                        output_tokens,\n                        input_tokens,\n                        first_64_token_logprobs,\n                    )\n',
        1,
    )
    openai_text = openai_text.replace(
        '                        output_tokens,\n                        input_tokens,\n                    ) = future.result()\n',
        '                        output_tokens,\n                        input_tokens,\n                        first_64_token_logprobs,\n                    ) = future.result()\n',
        1,
    )
    if 'payload["logprobs"] = True' not in openai_text:
        openai_text = openai_text.replace(
            '            if "top_p" in request_gen_kwargs:\n                payload["top_p"] = request_gen_kwargs.get("top_p")\n\n',
            '            if "top_p" in request_gen_kwargs:\n                payload["top_p"] = request_gen_kwargs.get("top_p")\n\n            payload["logprobs"] = True\n            payload["top_logprobs"] = 5\n\n',
            1,
        )
    if '"first_64_token_logprobs": first_64_token_logprobs,' not in openai_text:
        openai_text = openai_text.replace(
            '                            "reasoning_tokens": reasoning_tokens,\n                        }\n',
            '                            "reasoning_tokens": reasoning_tokens,\n                            "first_64_token_logprobs": first_64_token_logprobs,\n                        }\n',
            1,
        )
    openai_path.write_text(openai_text, encoding="utf-8")
    print(f"[INFO] Patched first-64-token logprobs capture: {openai_path}")


patch_first_64_logprobs_capture()
