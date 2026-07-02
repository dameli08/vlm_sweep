# VLM Sweep Automation

Runs vLLM + lmms-eval sweeps and writes final artifacts under the project root.

## What This Runs

For each selected model:
1. Start one vLLM server.
2. Run the configured benchmark sweeps.
3. Write live answer JSONL files after each response.
4. Append one result row per run to `results/all_results.jsonl`.
5. Stop the server and move to the next model.

Sweeps are not combinational. Each benchmark runs:
- one temperature sweep
- one top-p sweep
- one repetition-penalty sweep

The final configuration is thinking-mode only by default: `RUN_NON_REASON="0"`, `RUN_REASON="1"`.

## Benchmarks

Configured in [config.sh](config.sh):
- Reason: `mmmu_pro_vision_cot_reasoning`, `ai2d_reasoning`
- Non-reason entries remain available for compatibility, but are disabled by default.

## Setup

```bash
cd /home/dameli/vlm_sweep/code
chmod +x setup_env.sh run_vlm_benchmark_sweeps.sh
./setup_env.sh vlm_sweep_20260617
```

The setup script also applies the local lmms-eval patch for reasoning capture, strict MCQ prompting, and Qwen/Gemma reasoning-output splitting.

## Configure

Edit [config.sh](config.sh):
- `MODEL_PATHS`, `MODEL_ALIASES`, and `ANSWER_MODEL_NAMES`
- `SELECTED_MODEL_ALIASES` if you want to run only a subset of the 11 configured models
- GPU/server settings if needed
- `LIMIT` for smoke tests

No external judge API key is required.

## Run

```bash
cd /home/dameli/vlm_sweep/code
./run_vlm_benchmark_sweeps.sh ./config.sh
```

Final artifacts:
- `/home/dameli/vlm_sweep/answers/<model_name>/*.jsonl`
- `/home/dameli/vlm_sweep/results/all_results.jsonl`

Thinking-mode answer rows contain:
- `true_answer`
- `thinking_process`
- `response`

`response` is extracted locally as a single final option letter. If no clean final letter can be extracted, `response` is empty and the row is excluded from the accuracy denominator.

Each result row contains `accuracy`, `overall_questions`, and `answered_questions`.
