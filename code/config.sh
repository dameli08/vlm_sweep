#!/usr/bin/env bash

# =========================
# Environment
# =========================
CONDA_ENV_NAME="vlm_sweep_20260617"

# =========================
# vLLM server settings
# =========================
CUDA_VISIBLE_DEVICES="0"
HOST="127.0.0.1"
PORT="23333"
TENSOR_PARALLEL_SIZE="1"
GPU_MEMORY_UTILIZATION="0.95"
TRUST_REMOTE_CODE="1"
MAX_MODEL_LEN=""
MAX_NUM_SEQS=""
SERVER_START_TIMEOUT_SEC="240"

# =========================
# OpenAI-compatible endpoint settings
# =========================
OPENAI_API_KEY="EMPTY"
API_TIMEOUT="1200"
API_MAX_RETRIES="3"

# =========================
# Evaluator settings
# =========================
EVAL_MODEL_BACKEND="openai_compatible_chat"
MAX_NEW_TOKENS="1024"
NON_REASON_MAX_NEW_TOKENS="1024"
REASON_MAX_NEW_TOKENS="12000"
BATCH_SIZE="1"
LIMIT=""  # keep empty for full benchmark; set e.g. 10 for smoke tests
LOG_SAMPLES="1"
EXPORT_CSV="0"
RUN_NON_REASON="1"
RUN_REASON="1"

# Optional mode-specific extra model args.
# For Qwen3.5 thinking mode, these are commonly used with vLLM OpenAI endpoint.
NON_REASON_EXTRA_MODEL_ARGS='enable_thinking=false,force_letter_output=true'
REASON_EXTRA_MODEL_ARGS='enable_thinking=true,force_letter_output=true'

# Optional mode-specific extra generation kwargs appended to --gen_kwargs
NON_REASON_EXTRA_GEN_KWARGS=""
REASON_EXTRA_GEN_KWARGS=""

# =========================
# Benchmarks
# =========================
# Non-reason run uses these IDs.
NON_REASON_BENCHMARKS=(
  "mmmu_pro_vision"
  "ai2d"
  "mmstar"
)

# Reason run uses reasoning IDs.
REASON_BENCHMARKS=(
  "mmmu_pro_vision_cot_reasoning"
  "ai2d_reasoning"
  "mmstar_reasoning"
)

# =========================
# Models (test setup requested)
# =========================
# Keep these arrays same length and order.
MODEL_PATHS=(
  "/data/models/Qwen3.5-4B"
  "/data/models/Qwen3.5-9B"
)

MODEL_ALIASES=(
  "qwen3.5-4b"
  "qwen3.5-9b"
)

# Names used for output folders under answers/.
ANSWER_MODEL_NAMES=(
  "qwen_4b"
  "qwen_9b"
)

# =========================
# Sweep values (NO combinations)
# =========================
TEMPERATURE_VALUES=(0.2 0.4)
TOP_P_VALUES=(0.95 1.0)
REPETITION_PENALTY_VALUES=(1.10 1.30)

# Fixed architecture/generation value included in every run record.
TOP_K_VALUE="20"

# Mode-specific baselines used while sweeping one parameter at a time.
NON_REASON_BASELINE_TEMPERATURE="0.7"
NON_REASON_BASELINE_TOP_P="0.8"
NON_REASON_BASELINE_REPETITION_PENALTY="1.0"

REASON_BASELINE_TEMPERATURE="1.0"
REASON_BASELINE_TOP_P="0.95"
REASON_BASELINE_REPETITION_PENALTY="1.0"

# =========================
# Output
# =========================
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${CODE_DIR}/.." && pwd)"
ANSWERS_ROOT="${PROJECT_ROOT}/answers"
RESULTS_ROOT="${PROJECT_ROOT}/results"
RESULTS_JSONL="${RESULTS_ROOT}/all_results.jsonl"
WORK_ROOT="/tmp/vlm_sweep_work"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"

# Reason-mode answer extraction.
# Set GEMINI_API_KEY in your shell before running; do not store it in this repo.
GEMINI_API_KEY="${GEMINI_API_KEY:-}"
GEMINI_MODEL="gemini-3.1-flash-lite"
EXPORT_ARTIFACTS="1"

# Continue/stop all runs even if one experiment fails.
CONTINUE_ON_ERROR="0"
