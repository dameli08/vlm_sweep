#!/usr/bin/env bash
set -euo pipefail

cd /home/dameli/vlm_sweep

run_one() {
  local benchmark="$1"
  local rep="$2"
  local seed="$3"

  local cfg
  cfg="$(mktemp /home/dameli/vlm_sweep/code/rerun_cfg_XXXXXX.sh)"

  cp code/config.sh "$cfg"

  cat >> "$cfg" <<EOF

# ---- targeted rerun override ----
SELECTED_MODEL_ALIASES=("qwen3.5-4b")
RUN_NON_REASON="0"
RUN_REASON="1"
REASON_BENCHMARKS=("${benchmark}")
REPETITION_PENALTY_VALUES=(${rep})
PRESENCE_PENALTY_VALUES=()
SAMPLE_SEEDS=(${seed})
CONTINUE_ON_ERROR="1"
EXPORT_ARTIFACTS="1"
RUN_TAG="repair_${benchmark}_rep${rep}_seed${seed}_$(date +%Y%m%d_%H%M%S)"
EOF

  bash code/run_vlm_benchmark_sweeps.sh "$cfg"
}

run_one "mmmu_pro_vision_cot_reasoning" "1.15" "0"
run_one "mmmu_pro_vision_cot_reasoning" "1.15" "1"
run_one "mmmu_pro_vision_cot_reasoning" "1.2"  "2"
run_one "ai2d_reasoning"                 "1.2"  "1"
