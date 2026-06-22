# Fresh VLM Sweep Automation (vLLM + lmms-eval)

This folder is fully new and independent. No existing files/folders were modified.

## What this runs

For each model in order:
1. Start one vLLM server for that model.
2. Run all non-reason benchmarks with parameter sweeps.
3. Run all reason benchmarks with parameter sweeps.
4. Stop server and move to next model.

Sweeps are NOT combinational:
- Temperature sweep only
- Top-p sweep only
- Repetition-penalty sweep only

So per benchmark, per mode:
- `len(TEMPERATURE_VALUES) + len(TOP_P_VALUES) + len(REPETITION_PENALTY_VALUES)`

With the current test values (`2 + 2 + 2`), that is `6` runs per benchmark per mode.
For 3 benchmarks and 2 modes, one model has `3 * 6 * 2 = 36` runs.

## Benchmarks

Configured in [config.sh](config.sh):
- Non-reason: `mmmu_pro_vision`, `ocrbench`, `mmstar`
- Reason: `mmmu_pro_vision_cot_reasoning`, `ocrbench_reasoning`, `mmstar_reasoning`

## Do you need to manually download benchmarks?

Usually no.
`lmms-eval` typically downloads required benchmark data automatically on first run.

## Setup

```bash
cd /home/dameli/vlm_sweep/code
chmod +x setup_env.sh run_vlm_benchmark_sweeps.sh
./setup_env.sh vlm_sweep_20260617
```

The setup script pins `antlr4-python3-runtime==4.9.3` after installing `lmms-eval` because its legacy and extended LaTeX parser dependencies publish conflicting runtime metadata. Both parser paths are smoke-tested during setup.

## Configure

Edit [config.sh](config.sh):
- `MODEL_PATHS`, `MODEL_ALIASES`, and `ANSWER_MODEL_NAMES`
- Sweep values: temperature, top-p, repetition penalty
- Separate non-thinking and thinking baseline values: `NON_REASON_BASELINE_*` and `REASON_BASELINE_*`
- Fixed `TOP_K_VALUE`
- `EXPORT_CSV="0"` to discard temporary CSV files after each run
- `GEMINI_API_KEY` from your shell environment for OCRBench judging
- GPU/server settings if needed

Your current test models are already set to:
- `/data/models/Qwen3.5-4B`
- `/data/models/Qwen3.5-9B`

## Run

Set the Gemini key only in your shell, not in any tracked file:

```bash
export GEMINI_API_KEY="your_key_here"
cd /home/dameli/vlm_sweep/code
./run_vlm_benchmark_sweeps.sh ./config.sh
```

Final artifacts are stored under:
- `/home/dameli/vlm_sweep/answers/<model_name>/*.jsonl`
- `/home/dameli/vlm_sweep/results/all_results.jsonl`

Answer JSONL files are updated after each completed model response. Non-thinking rows contain:
- `true_answer`
- `response`

Thinking-mode rows contain the complete original generation in `thinking_process`. Gemini extracts the explicit final answer for MMMU Pro, MMStar, and OCRBench into `response`; it is not given the ground-truth answer during extraction. A missing explicit final answer produces an empty `response`, which is excluded from accuracy.

The consolidated results JSONL stores one record per benchmark sweep run with the run parameters, fixed top-k value, accuracy, and valid/invalid row counts. Empty or invalid answers are excluded from the accuracy denominator. Temporary lmms-eval logs and raw outputs are written under `WORK_ROOT` from [config.sh](config.sh), currently `/tmp/vlm_sweep_work`.
