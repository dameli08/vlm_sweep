# VLM Sweep Automation

Runs vLLM + lmms-eval vision sweeps for the penalty project.

## Current Protocol

- Reasoning mode only by default: `RUN_NON_REASON="0"`, `RUN_REASON="1"`.
- Benchmarks: MMMU-Pro vision, AI2D, and MathVision testmini.
- Fixed item sets: 250 seeded-random MMMU-Pro rows, 250 seeded-random AI2D rows, and all 304 MathVision testmini rows.
- Samples per item: k=5 explicit seed runs, configured by `SAMPLE_SEEDS`.
- Sweeps are separate 1-D sweeps:
  - `repetition_penalty=(1.0 1.05 1.1 1.15 1.2)` with `presence_penalty=0`
  - `presence_penalty=(0 0.5 1.0 1.5 2.0)` with `repetition_penalty=1.0`
- Temperature and top-p are fixed at the configured vendor values: `temperature=1.0`, `top_p=0.95`.

## Setup

```bash
cd /home/dameli/vlm_sweep/code
chmod +x setup_env.sh run_vlm_benchmark_sweeps.sh
./setup_env.sh vlm_sweep_20260617
```

The runner applies the local lmms-eval patch at startup. That patch handles strict reasoning prompts, Qwen/Gemma reasoning-output splitting, MathVision prompting, fixed item subsets, and forwarding `presence_penalty` plus per-run `seed` to vLLM.

## Configure

Edit [config.sh](config.sh):

- `MODEL_PATHS`, `MODEL_ALIASES`, and `ANSWER_MODEL_NAMES`
- `SELECTED_MODEL_ALIASES` to choose the models that fit on the current GPU
- GPU/server settings if needed
- `SUBSAMPLE_SEED` only if you intentionally want a different fixed item set

No Gemini or other judge API key is used.

## Run

```bash
cd /home/dameli/vlm_sweep/code
./run_vlm_benchmark_sweeps.sh ./config.sh
```

Final artifacts:

- `/home/dameli/vlm_sweep/answers/<model_name>/*.jsonl`
- `/home/dameli/vlm_sweep/results/all_results.jsonl`
- `/home/dameli/vlm_sweep/code/subsamples/vision_subsample_seed_20260710.json`

Reasoning answer rows contain `item_id`, `seed`, `true_answer`, `thinking_process`, `raw_output`, and `response`. They also include lightweight generation metadata: `finish_reason`, `budget_truncation_flag`, `trace_token_count`, `answer_token_count`, `loop_score`, `max_tokens`, `thinking_budget`, `model_id`, `model_path`, `vllm_version`, `harness_git_commit`, `timestamp`, `request_success`, `input_tokens`, `output_tokens`, and `reasoning_tokens`.

Result rows include per-seed rows and aggregate rows. They also include `model_id`, `model_path`, `max_tokens`, `thinking_budget`, `vllm_version`, `harness_git_commit`, and `timestamp`.

Per-seed rows report accuracy over answered/parseable questions only. Unparseable outputs are counted in `format_failure_count` and excluded from that per-seed accuracy denominator.

Aggregate rows use strict majority vote across the five seeds for each item. Unparseable outputs do not vote, so they can prevent a 3-of-5 majority; items without a correct strict majority count as wrong in aggregate accuracy. In aggregate rows, `answered_questions` is the number of items with a valid strict majority answer.

Note: full top-k token logprobs and entropy metrics are still not collected.
