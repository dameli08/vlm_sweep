#!/usr/bin/env bash
set -euo pipefail

# Isolated reasoning-mode test using the same lmms-eval backend path as the main runner.
# This does NOT modify run_vlm_benchmark_sweeps.sh and writes to its own output folder.
#
# Usage:
#   bash test_reasoning_isolated.sh [model_alias] [limit]
# Example:
#   bash test_reasoning_isolated.sh qwen3.5-4b 3

MODEL_ALIAS="${1:-qwen3.5-4b}"
LIMIT="${2:-3}"
BASE_URL="http://127.0.0.1:23333/v1"
OUT_DIR="/home/dameli/vlm_sweep/reason_isolated_$(date +%Y%m%d_%H%M%S)"
TASK_CANDIDATES=(
  "mmmu_pro_vision_cot_reasoning"
  "mmstar_reasoning"
  "ocrbench_reasoning"
)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm_sweep_20260617

# Fast health check: make sure vLLM is up.
if ! curl -sS "${BASE_URL}/models" >/dev/null 2>&1; then
  echo "[ERROR] vLLM server is not reachable at ${BASE_URL}."
  echo "Start it first (or run your main sweep script, which starts it automatically)."
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "[INFO] Running isolated reasoning check"
echo "[INFO] model=${MODEL_ALIAS} limit=${LIMIT} out=${OUT_DIR}"

# Self-heal missing lmms-eval include files (same idea as main runner).
python - <<'PY'
from pathlib import Path
import importlib.util
import re

spec = importlib.util.find_spec("lmms_eval")
if spec is None or spec.origin is None:
  raise SystemExit("lmms_eval is not importable")

tasks_root = Path(spec.origin).resolve().parent / "tasks"
fallback_root = Path.home() / "miniconda3" / "envs" / "vllmserve" / "lib" / "python3.10" / "site-packages" / "lmms_eval" / "tasks"
pat = re.compile(r"^\s*[\"\']?include[\"\']?\s*:\s*([^\n#]+)", flags=re.M)

def scan_files():
  out = []
  for p in tasks_root.rglob("*"):
    if not p.is_file():
      continue
    if p.suffix in (".py", ".so", ".pyd", ".pyc"):
      continue
    out.append(p)
  return out

created = 0
for _ in range(12):
  step = 0
  for path in scan_files():
    text = path.read_text(encoding="utf-8", errors="ignore")
    for m in pat.finditer(text):
      include_name = m.group(1).strip().strip('"\'').split()[0]
      if not include_name or include_name.startswith("http"):
        continue
      target = path.parent / include_name
      if target.exists():
        continue
      candidates = [
        path.parent / f"{include_name}.yaml",
        fallback_root / target.relative_to(tasks_root),
        fallback_root / target.relative_to(tasks_root).with_suffix(".yaml"),
      ]
      source = next((c for c in candidates if c.exists()), None)
      if source is None:
        continue
      target.parent.mkdir(parents=True, exist_ok=True)
      target.write_text(source.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
      step += 1
  created += step
  if step == 0:
    break

print(f"[INFO] lmms include repair created={created}")
PY

run_one_mode() {
  local mode_name="$1"
  local force_letter="$2"
  local out_subdir="$3"

  local task
  for task in "${TASK_CANDIDATES[@]}"; do
  local out_path="${OUT_DIR}/${out_subdir}"
  local log_path="${OUT_DIR}/${out_subdir}.log"
  mkdir -p "${out_path}"

  echo "[INFO] ${mode_name}: trying task=${task}" | tee -a "${log_path}"

  if lmms-eval \
    --model openai_compatible_chat \
    --model_args "model=${MODEL_ALIAS},base_url=${BASE_URL},api_key=EMPTY,timeout=600,max_retries=5,enable_thinking=true,force_letter_output=${force_letter}" \
    --tasks "${task}" \
    --gen_kwargs "temperature=0.2,top_p=0.95,max_new_tokens=12000" \
    --batch_size 1 \
    --limit "${LIMIT}" \
    --log_samples \
    --output_path "${out_path}" >>"${log_path}" 2>&1; then
    echo "[INFO] ${mode_name}: success with task=${task}" | tee -a "${log_path}"
    echo "${task}" >"${out_path}/task_used.txt"
    return 0
  fi

  echo "[WARN] ${mode_name}: task failed -> ${task}. Trying next..." | tee -a "${log_path}"
  done

  echo "[ERROR] ${mode_name}: all reasoning task candidates failed." | tee -a "${OUT_DIR}/${out_subdir}.log"
  return 1
}

# 1) Strict mode: exactly like your runner's reason mode behavior.
run_one_mode "strict" "true" "strict" || true

# 2) Visibility mode: same reasoning setup, but disable forced letter normalization
# so you can inspect whether <think> appears in raw response text.
run_one_mode "visibility" "false" "visible" || true

# Quick hints for where to inspect outputs.
echo ""
echo "[DONE] Isolated reasoning tests finished."
echo "Strict-mode output:   ${OUT_DIR}/strict"
echo "Visibility-mode output: ${OUT_DIR}/visible"
echo "Logs: ${OUT_DIR}/strict.log and ${OUT_DIR}/visible.log"
echo ""
echo "To check whether thinking text is present in visibility mode:"
echo "grep -RIn '<think>\|</think>' '${OUT_DIR}/visible' || true"
