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
RUN_NON_REASON="0"
RUN_REASON="1"

# Optional mode-specific extra model args.
NON_REASON_EXTRA_MODEL_ARGS='enable_thinking=false,force_letter_output=true'
REASON_EXTRA_MODEL_ARGS='enable_thinking=true,force_letter_output=true'

# Optional mode-specific extra generation kwargs appended to --gen_kwargs
NON_REASON_EXTRA_GEN_KWARGS=""
REASON_EXTRA_GEN_KWARGS=""

# =========================
# Benchmarks
# =========================
NON_REASON_BENCHMARKS=(
  "mmmu_pro_vision"
  "ai2d"
)

REASON_BENCHMARKS=(
  "mmmu_pro_vision_cot_reasoning"
  "ai2d_reasoning"
)

# =========================
# Models
# =========================
# Keep these arrays same length and order. Edit paths to match local storage.
MODEL_PATHS=(
  "/data/models/Qwen3.5-4B"
  "/data/models/Qwen3.5-9B"
  "/data/models/Qwen3.5-27B"
  "/data/models/Qwen3.6-27B"
  "/data/models/Qwen3.6-35B"
  "/data/models/gemma-4-E2B-it"
  "/data/models/gemma-4-4b-it"
  "/data/models/gemma-4-12B-it"
  "/data/models/Gemma4-26B"
  "/data/models/Gemma4-32B"
  "Qwen/Qwen3-Omni-30B-A3B-Thinking"
)

MODEL_ALIASES=(
  "qwen3.5-4b"
  "qwen3.5-9b"
  "qwen3.5-27b"
  "qwen3.6-27b"
  "qwen3.6-35b"
  "gemma4-2b"
  "gemma4-4b"
  "gemma4-12b"
  "gemma4-26b"
  "gemma4-32b"
  "qwen3-omni-30b-a3b-thinking"
)

ANSWER_MODEL_NAMES=(
  "qwen3.5_4b"
  "qwen3.5_9b"
  "qwen3.5_27b"
  "qwen3.6_27b"
  "qwen3.6_35b"
  "gemma4_2b"
  "gemma4_4b"
  "gemma4_12b"
  "gemma4_26b"
  "gemma4_32b"
  "qwen3_omni_30b_a3b_thinking"
)

# Leave empty to run every model above. To run a subset, list aliases here, e.g.:
# SELECTED_MODEL_ALIASES=("qwen3.5-4b" "gemma4-2b")
SELECTED_MODEL_ALIASES=("qwen3.5-4b" "qwen3.5-9b" "gemma4-2b" "gemma4-4b" "gemma4-12b")

# =========================
# Sweep values (NO combinations)
# =========================
TEMPERATURE_VALUES=(0.8 0.9 1.0)
TOP_P_VALUES=(0.85 0.90 0.95)
REPETITION_PENALTY_VALUES=(1.0 1.1 1.2)

TOP_K_VALUE="20"

#NON_REASON_BASELINE_TEMPERATURE="0.8"
#NON_REASON_BASELINE_TOP_P="0.90"
#NON_REASON_BASELINE_REPETITION_PENALTY="1.0"

REASON_BASELINE_TEMPERATURE="1.0"
REASON_BASELINE_TOP_P="0.95"
REASON_BASELINE_REPETITION_PENALTY="1.0"

# Optional per-model reasoning baselines. Keys must match MODEL_ALIASES.
# If a model alias is not listed, the global REASON_BASELINE_* values above are used.
declare -A REASON_BASELINE_TEMPERATURE_BY_MODEL=(
  ["qwen3.5-4b"]="1.0"
  ["qwen3.5-9b"]="1.0"
  ["gemma4-2b"]="1.0"
  ["gemma4-4b"]="1.0"
  ["gemma4-12b"]="1.0"
)

declare -A REASON_BASELINE_TOP_P_BY_MODEL=(
  ["qwen3.5-4b"]="0.95"
  ["qwen3.5-9b"]="0.95"
  ["gemma4-2b"]="0.95"
  ["gemma4-4b"]="0.95"
  ["gemma4-12b"]="0.95"
)

declare -A REASON_BASELINE_REPETITION_PENALTY_BY_MODEL=(
  ["qwen3.5-4b"]="1.0"
  ["qwen3.5-9b"]="1.0"
  ["gemma4-2b"]="1.0"
  ["gemma4-4b"]="1.0"
  ["gemma4-12b"]="1.0"
)

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

EXPORT_ARTIFACTS="1"
CONTINUE_ON_ERROR="0"
