#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/config.sh}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG_PATH}"
  exit 1
fi

# shellcheck source=/dev/null
source "${CONFIG_PATH}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
python "${SCRIPT_DIR}/patch_lmms_reasoning_capture.py"

if ! command -v vllm >/dev/null 2>&1; then
  echo "[ERROR] vllm not found in env ${CONDA_ENV_NAME}. Run setup_env.sh first."
  exit 1
fi
if ! command -v lmms-eval >/dev/null 2>&1; then
  echo "[ERROR] lmms-eval not found in env ${CONDA_ENV_NAME}. Run setup_env.sh first."
  exit 1
fi

# Self-heal for known lmms-eval packaging issues where include files
# referenced by task YAMLs may be missing.
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
  files = []
  for p in tasks_root.rglob("*"):
    if not p.is_file():
      continue
    if p.suffix in (".py", ".so", ".pyd", ".pyc"):
      continue
    files.append(p)
  return files

created = 0
for _ in range(12):
  step_created = 0
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
      step_created += 1

  created += step_created
  if step_created == 0:
    break

unresolved = set()
for path in scan_files():
  text = path.read_text(encoding="utf-8", errors="ignore")
  for m in pat.finditer(text):
    include_name = m.group(1).strip().strip('"\'').split()[0]
    if not include_name or include_name.startswith("http"):
      continue
    target = path.parent / include_name
    if not target.exists():
      unresolved.add(str(target.relative_to(tasks_root)))

print(f"[INFO] lmms-eval include repair: created={created}, unresolved={len(unresolved)}")
PY

if [[ ${#MODEL_PATHS[@]} -ne ${#MODEL_ALIASES[@]} ]]; then
  echo "[ERROR] MODEL_PATHS and MODEL_ALIASES must have same length."
  exit 1
fi
if [[ ${#MODEL_PATHS[@]} -ne ${#ANSWER_MODEL_NAMES[@]} ]]; then
  echo "[ERROR] MODEL_PATHS and ANSWER_MODEL_NAMES must have same length."
  exit 1
fi

mkdir -p "${ANSWERS_ROOT}" "${RESULTS_ROOT}" "${WORK_ROOT}"
RUN_ROOT="${WORK_ROOT}/${RUN_TAG}"
mkdir -p "${RUN_ROOT}"
touch "${RESULTS_JSONL}"

SUBSAMPLE_PATH="${SUBSAMPLE_ROOT}/vision_subsample_seed_${SUBSAMPLE_SEED}.json"
mkdir -p "${SUBSAMPLE_ROOT}"
if [[ ! -f "${SUBSAMPLE_PATH}" ]]; then
  mathvision_size_arg="0"
  if [[ -n "${SUBSAMPLE_SIZE_MATHVISION}" ]]; then
    mathvision_size_arg="${SUBSAMPLE_SIZE_MATHVISION}"
  fi
  python "${SCRIPT_DIR}/prepare_subsamples.py" \
    --out "${SUBSAMPLE_PATH}" \
    --seed "${SUBSAMPLE_SEED}" \
    --mmmu-pro-size "${SUBSAMPLE_SIZE_MMMU_PRO}" \
    --ai2d-size "${SUBSAMPLE_SIZE_AI2D}" \
    --mathvision-size "${mathvision_size_arg}"
fi
export LMMS_FIXED_SUBSET_PATH="${SUBSAMPLE_PATH}"
echo "[INFO] Fixed subset path: ${LMMS_FIXED_SUBSET_PATH}"

OPENAI_BASE_URL="http://${HOST}:${PORT}/v1"
VLLM_VERSION="$(python -c 'import importlib.metadata as m; print(m.version("vllm"))' 2>/dev/null || echo "")"
HARNESS_GIT_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo "")"
TOTAL_OK=0
TOTAL_FAIL=0
SERVER_PID=""
LIVE_WRITER_PID=""
LIVE_WRITER_STOP_FILE=""

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

cleanup() {
  if [[ -n "${LIVE_WRITER_PID}" ]] && kill -0 "${LIVE_WRITER_PID}" >/dev/null 2>&1; then
    [[ -n "${LIVE_WRITER_STOP_FILE}" ]] && touch "${LIVE_WRITER_STOP_FILE}"
    wait "${LIVE_WRITER_PID}" >/dev/null 2>&1 || true
  fi
  LIVE_WRITER_PID=""
  LIVE_WRITER_STOP_FILE=""

  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    log "Stopping vLLM server PID ${SERVER_PID}"
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local timeout_sec="$1"
  local start_epoch
  start_epoch="$(date +%s)"

  while true; do
    if curl -sS "${OPENAI_BASE_URL}/models" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      return 1
    fi
    local now
    now="$(date +%s)"
    if (( now - start_epoch > timeout_sec )); then
      return 1
    fi
    sleep 2
  done
}

start_server() {
  local model_path="$1"
  local model_alias="$2"
  local server_log="$3"

  export CUDA_VISIBLE_DEVICES

  local -a cmd
  cmd=(
    vllm serve "${model_path}"
    --host "${HOST}"
    --port "${PORT}"
    --served-model-name "${model_alias}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  )

  if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    cmd+=(--trust-remote-code)
  fi
  if [[ -n "${MAX_MODEL_LEN}" ]]; then
    cmd+=(--max-model-len "${MAX_MODEL_LEN}")
  fi
  if [[ -n "${MAX_NUM_SEQS}" ]]; then
    cmd+=(--max-num-seqs "${MAX_NUM_SEQS}")
  fi

  log "Starting vLLM server for ${model_alias}"
  nohup "${cmd[@]}" >"${server_log}" 2>&1 &
  SERVER_PID="$!"

  if wait_for_server "${SERVER_START_TIMEOUT_SEC}"; then
    log "vLLM server is ready for ${model_alias}"
    return 0
  fi

  log "[ERROR] vLLM server failed to start for ${model_alias}. See ${server_log}"
  return 1
}

build_model_args() {
  local model_alias="$1"
  local mode="$2"
  local benchmark="$3"

  local model_args="model=${model_alias},base_url=${OPENAI_BASE_URL},api_key=${OPENAI_API_KEY},timeout=${API_TIMEOUT},max_retries=${API_MAX_RETRIES}"

  if [[ "${mode}" == "non_reason" ]]; then
    if [[ -n "${NON_REASON_EXTRA_MODEL_ARGS}" ]]; then
      model_args+=" ,${NON_REASON_EXTRA_MODEL_ARGS}"
    fi
  else
    if [[ -n "${REASON_EXTRA_MODEL_ARGS}" ]]; then
      model_args+=" ,${REASON_EXTRA_MODEL_ARGS}"
    fi
  fi

  # Use benchmark-specific answer format policy:
  # - MMMU Pro and AI2D are MCQ => force single-letter output
  model_args="$(echo "${model_args}" | sed -E 's/,?force_letter_output=[^,]*//g; s/,,+/,/g; s/,$//')"
  if [[ "${benchmark}" == *"mmmu"* || "${benchmark}" == *"ai2d"* ]]; then
    model_args+=",force_letter_output=true"
  else
    model_args+=",force_letter_output=false"
  fi

  # Remove accidental spaces before commas.
  echo "${model_args}" | sed 's/ ,/,/g'
}

build_gen_kwargs() {
  local mode="$1"
  local temperature="$2"
  local top_p="$3"
  local repetition_penalty="$4"
  local presence_penalty="$5"
  local sample_seed="$6"
  local max_tokens

  if [[ "${mode}" == "non_reason" ]]; then
    max_tokens="${NON_REASON_MAX_NEW_TOKENS:-${MAX_NEW_TOKENS}}"
  else
    max_tokens="${REASON_MAX_NEW_TOKENS:-${MAX_NEW_TOKENS}}"
  fi

  local gen_kwargs="temperature=${temperature},top_p=${top_p},top_k=${TOP_K_VALUE},repetition_penalty=${repetition_penalty},presence_penalty=${presence_penalty},seed=${sample_seed},max_new_tokens=${max_tokens}"

  if [[ "${mode}" == "non_reason" ]]; then
    if [[ -n "${NON_REASON_EXTRA_GEN_KWARGS}" ]]; then
      gen_kwargs+=",${NON_REASON_EXTRA_GEN_KWARGS}"
    fi
  else
    if [[ -n "${REASON_EXTRA_GEN_KWARGS}" ]]; then
      gen_kwargs+=",${REASON_EXTRA_GEN_KWARGS}"
    fi
  fi

  echo "${gen_kwargs}"
}

export_run_csvs() {
  local out_dir="$1"
  local benchmark="$2"
  local mode="$3"

  python - "$out_dir" "$benchmark" "$mode" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
benchmark = sys.argv[2]
mode = sys.argv[3]

results_files = sorted(out_dir.rglob("*_results.json"))
if results_files:
    results_path = results_files[0]
    data = json.loads(results_path.read_text(encoding="utf-8"))
    task_results = data.get("results", {}).get(benchmark, {})

    metrics_csv = out_dir / "metrics_table.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerow(["benchmark", "metric", "value"])
      for k, v in task_results.items():
        if k == "alias":
          continue
        writer.writerow([benchmark, k, v])

sample_candidates = []
for p in out_dir.rglob("*"):
    if not p.is_file():
      continue
    name = p.name.lower()
    if "sample" not in name:
      continue
    if p.suffix.lower() not in (".json", ".jsonl"):
      continue
    sample_candidates.append(p)

rows = []
live_rows = {}

live_csv = out_dir / "answers_live.csv"
if live_csv.exists():
  try:
    with live_csv.open(encoding="utf-8", newline="") as f:
      for rec in csv.DictReader(f):
        key = (str(rec.get("benchmark", "")), str(rec.get("doc_id", "")))
        live_rows[key] = rec
  except Exception:
    live_rows = {}

def extract_final_letter(text):
  if text is None:
    return ""
  if not isinstance(text, str):
    text = json.dumps(text, ensure_ascii=False)
  upper = text.upper()
  patterns = [
    r"FINAL\s*ANSWER\s*[:=\-]?\s*\(?\s*([A-Z])\s*\)?",
    r"FINAL\s*ANSWER\s*(?:IS)?\s*\(?\s*([A-Z])\s*\)?",
    r"ANSWER\s*[:=\-]?\s*\(?\s*([A-Z])\s*\)?",
    r"OPTION\s*[:=\-]?\s*\(?\s*([A-Z])\s*\)?",
  ]
  import re
  for pat in patterns:
    m = re.findall(pat, upper)
    if m:
      return m[-1]
  tail = upper[-60:]
  tokens = re.findall(r"\b([A-Z])\b", tail)
  if tokens:
    return tokens[-1]
  return ""

def strip_think_blocks(text):
  if not isinstance(text, str):
    return text
  import re
  cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
  cleaned = re.sub(r"<analysis>.*?</analysis>", "", cleaned, flags=re.I | re.S)
  return cleaned.strip()

def normalize_exact_text(text):
  if text is None:
    return ""
  if not isinstance(text, str):
    text = json.dumps(text, ensure_ascii=False)
  s = text.strip()
  try:
    import ast
    parsed = ast.literal_eval(s)
    if isinstance(parsed, list):
      if len(parsed) == 1:
        s = str(parsed[0])
      else:
        s = " ".join(str(x) for x in parsed)
    elif isinstance(parsed, tuple):
      s = " ".join(str(x) for x in parsed)
  except Exception:
    pass
  import re
  s = re.sub(r"[\[\]\(\)\{\}\"']", "", s)
  s = re.sub(r"\s+", " ", s).strip().lower()
  return s

def benchmark_match_rule(benchmark_name):
  b = (benchmark_name or "").lower()
  return "letter"

def to_text(x):
    if x is None:
      return ""
    if isinstance(x, str):
      return x
    return json.dumps(x, ensure_ascii=False)

def pick_prediction(rec):
    keys = ["prediction", "pred", "response", "resps", "filtered_resps", "model_output", "model_outputs"]
    for k in keys:
      if k in rec and rec[k] is not None:
        return rec[k]
    return None

def pick_target(rec):
    if "target" in rec:
      return rec["target"]
    doc = rec.get("doc")
    if isinstance(doc, dict):
      for k in ("answer", "target", "label"):
        if k in doc:
          return doc[k]
    return None

def pick_doc_id(rec):
    for k in ("doc_id", "id", "instance_id"):
      if k in rec:
        return rec[k]
    doc = rec.get("doc")
    if isinstance(doc, dict):
      for k in ("id", "question_id", "pid"):
        if k in doc:
          return doc[k]
    return None

for p in sorted(sample_candidates):
    if p.suffix.lower() == ".jsonl":
      for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
          continue
        try:
          rec = json.loads(line)
        except Exception:
          continue
        pred_raw = to_text(pick_prediction(rec))
        pred_clean = strip_think_blocks(pred_raw) if mode == "non_reason" else pred_raw
        tgt_text = to_text(pick_target(rec))
        live = live_rows.get((benchmark, str(pick_doc_id(rec))), {})
        live_raw = to_text(live.get("raw_response_full", ""))
        live_visible = to_text(live.get("visible_response", ""))
        if not live_visible:
          live_visible = pred_clean
        rule = benchmark_match_rule(benchmark)
        if rule == "letter":
          true_letter = extract_final_letter(tgt_text)
          pred_letter = extract_final_letter(pred_clean)
        else:
          true_letter = ""
          pred_letter = ""
        letter_match = ""
        if true_letter and pred_letter:
          letter_match = "1" if true_letter == pred_letter else "0"
        true_norm = normalize_exact_text(tgt_text)
        if rule == "letter":
          pred_norm = normalize_exact_text(pred_clean)
        else:
          pred_norm = normalize_exact_text(live_visible)
        exact_match = ""
        if true_norm and pred_norm:
          exact_match = "1" if true_norm == pred_norm else "0"
        resolved_match = letter_match if rule == "letter" else exact_match
        rows.append([
          benchmark,
          pick_doc_id(rec),
          tgt_text,
          pred_raw,
          pred_clean,
          live_raw,
          to_text(live.get("thinking_trace", "")),
          live_visible,
          true_letter,
          pred_letter,
          letter_match,
          rule,
          true_norm,
          pred_norm,
          exact_match,
          resolved_match,
          to_text(rec),
        ])
    else:
      try:
        payload = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
      except Exception:
        continue
      if isinstance(payload, list):
        it = payload
      elif isinstance(payload, dict):
        if isinstance(payload.get("samples"), list):
          it = payload["samples"]
        else:
          it = []
      else:
        it = []
      for rec in it:
        if not isinstance(rec, dict):
          continue
        pred_raw = to_text(pick_prediction(rec))
        pred_clean = strip_think_blocks(pred_raw) if mode == "non_reason" else pred_raw
        tgt_text = to_text(pick_target(rec))
        live = live_rows.get((benchmark, str(pick_doc_id(rec))), {})
        live_raw = to_text(live.get("raw_response_full", ""))
        live_visible = to_text(live.get("visible_response", ""))
        if not live_visible:
          live_visible = pred_clean
        rule = benchmark_match_rule(benchmark)
        if rule == "letter":
          true_letter = extract_final_letter(tgt_text)
          pred_letter = extract_final_letter(pred_clean)
        else:
          true_letter = ""
          pred_letter = ""
        letter_match = ""
        if true_letter and pred_letter:
          letter_match = "1" if true_letter == pred_letter else "0"
        true_norm = normalize_exact_text(tgt_text)
        if rule == "letter":
          pred_norm = normalize_exact_text(pred_clean)
        else:
          pred_norm = normalize_exact_text(live_visible)
        exact_match = ""
        if true_norm and pred_norm:
          exact_match = "1" if true_norm == pred_norm else "0"
        resolved_match = letter_match if rule == "letter" else exact_match
        rows.append([
          benchmark,
          pick_doc_id(rec),
          tgt_text,
          pred_raw,
          pred_clean,
          live_raw,
          to_text(live.get("thinking_trace", "")),
          live_visible,
          true_letter,
          pred_letter,
          letter_match,
          rule,
          true_norm,
          pred_norm,
          exact_match,
          resolved_match,
          to_text(rec),
        ])

if not rows and live_rows:
    for rec in live_rows.values():
      rows.append([
        benchmark,
        rec.get("doc_id", ""),
        rec.get("true_answer_text", ""),
        rec.get("prediction_raw", ""),
        rec.get("prediction_clean", ""),
        rec.get("raw_response_full", ""),
        rec.get("thinking_trace", ""),
        rec.get("visible_response", ""),
        rec.get("true_letter", ""),
        rec.get("predicted_letter", ""),
        rec.get("letter_match", ""),
        rec.get("match_rule", ""),
        rec.get("true_answer_normalized", ""),
        rec.get("prediction_normalized", ""),
        rec.get("exact_match", ""),
        rec.get("resolved_match", ""),
        json.dumps(rec, ensure_ascii=False),
      ])

if rows:
    answers_csv = out_dir / "answers_raw.csv"
    with answers_csv.open("w", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerow(["benchmark", "doc_id", "true_answer_text", "prediction_raw", "prediction_clean", "raw_response_full", "thinking_trace", "visible_response", "true_letter", "predicted_letter", "letter_match", "match_rule", "true_answer_normalized", "prediction_normalized", "exact_match", "resolved_match", "record_json"])
      writer.writerows(rows)
PY
}

export_run_artifacts() {
  local out_dir="$1"
  local model_name="$2"
  local mode="$3"
  local benchmark="$4"
  local sweep_param="$5"
  local sweep_value="$6"
  local temperature="$7"
  local top_p="$8"
  local repetition_penalty="$9"
  local presence_penalty="${10}"
  local sample_seed="${11}"
  local model_alias="${12}"
  local model_path="${13}"
  local max_tokens="${14}"
  local thinking_budget="${15}"
  local sample_seed_csv
  sample_seed_csv="$(IFS=,; echo "${SAMPLE_SEEDS[*]}")"

  python "${SCRIPT_DIR}/export_artifacts.py" \
    --out-dir "${out_dir}" \
    --answers-root "${ANSWERS_ROOT}" \
    --results-jsonl "${RESULTS_JSONL}" \
    --model-name "${model_name}" \
    --mode "${mode}" \
    --benchmark "${benchmark}" \
    --sweep-param "${sweep_param}" \
    --sweep-value "${sweep_value}" \
    --temperature "${temperature}" \
    --top-p "${top_p}" \
    --top-k "${TOP_K_VALUE}" \
    --repetition-penalty "${repetition_penalty}" \
    --presence-penalty "${presence_penalty}" \
    --seed "${sample_seed}" \
    --sample-seeds "${sample_seed_csv}" \
    --model-id "${model_alias}" \
    --model-path "${model_path}" \
    --max-tokens "${max_tokens}" \
    --thinking-budget "${thinking_budget}" \
    --vllm-version "${VLLM_VERSION}" \
    --harness-git-commit "${HARNESS_GIT_COMMIT}" \
    --subset-path "${SUBSAMPLE_PATH}"
}

answer_benchmark_key() {
  local benchmark="${1,,}"
  if [[ "${benchmark}" == *"mmmu"* ]]; then
    echo "mmmu_pro"
  elif [[ "${benchmark}" == *"ai2d"* ]]; then
    echo "ai2d"
  elif [[ "${benchmark}" == *"mathvision"* ]]; then
    echo "mathvision_testmini"
  else
    echo "${benchmark}" | sed -E "s/[^a-z0-9]+/_/g; s/^_+|_+$//g"
  fi
}

stop_live_writer() {
  if [[ -n "${LIVE_WRITER_PID}" ]]; then
    touch "${LIVE_WRITER_STOP_FILE}"
    wait "${LIVE_WRITER_PID}" >/dev/null 2>&1 || true
    LIVE_WRITER_PID=""
    LIVE_WRITER_STOP_FILE=""
  fi
}

cleanup_run_csvs() {
  local out_dir="$1"

  if [[ "${EXPORT_CSV:-0}" == "0" ]]; then
    rm -f \
      "${out_dir}/answers_live.csv" \
      "${out_dir}/answers_raw.csv" \
      "${out_dir}/metrics_table.csv"
  fi
}

run_one_experiment() {
  local model_alias="$1"
  local answer_model_name="$2"
  local mode="$3"
  local benchmark="$4"
  local sweep_param="$5"
  local sweep_value="$6"
  local temperature="$7"
  local top_p="$8"
  local repetition_penalty="$9"
  local presence_penalty="${10}"
  local sample_seed="${11}"
  local model_path="${12}"
  local run_max_tokens
  if [[ "${mode}" == "non_reason" ]]; then
    run_max_tokens="${NON_REASON_MAX_NEW_TOKENS:-${MAX_NEW_TOKENS}}"
  else
    run_max_tokens="${REASON_MAX_NEW_TOKENS:-${MAX_NEW_TOKENS}}"
  fi
  local thinking_budget="${run_max_tokens}"

  local out_dir="${RUN_ROOT}/${model_alias}/${mode}/${benchmark}/${sweep_param}_${sweep_value}/seed_${sample_seed}"
  mkdir -p "${out_dir}"

  local marker_done="${out_dir}/.done"
  local marker_fail="${out_dir}/.failed"
  local run_log="${out_dir}/run.log"

  if [[ -f "${marker_done}" ]]; then
    log "SKIP done: ${model_alias} ${mode} ${benchmark} ${sweep_param}=${sweep_value}"
    return 0
  fi

  local model_args
  model_args="$(build_model_args "${model_alias}" "${mode}" "${benchmark}")"

  local gen_kwargs
  gen_kwargs="$(build_gen_kwargs "${mode}" "${temperature}" "${top_p}" "${repetition_penalty}" "${presence_penalty}" "${sample_seed}")"

  local live_answers_csv="${out_dir}/answers_live.csv"
  cat >"${live_answers_csv}" <<'CSV'
benchmark,mode,model_alias,doc_id,true_answer_text,prediction_raw,prediction_clean,raw_response_full,thinking_trace,visible_response,true_letter,predicted_letter,letter_match,match_rule,true_answer_normalized,prediction_normalized,exact_match,resolved_match,finish_reason,request_success,input_tokens,output_tokens,reasoning_tokens
CSV

  local answer_key
  answer_key="$(answer_benchmark_key "${benchmark}")"
  local safe_sweep_value
  safe_sweep_value="$(echo "${sweep_value}" | sed -E "s/[^A-Za-z0-9_.-]+/_/g")"
  local live_answer_jsonl="${ANSWERS_ROOT}/${answer_model_name}/${answer_key}__${mode}__${sweep_param}_${safe_sweep_value}__seed_${sample_seed}.jsonl"
  LIVE_WRITER_STOP_FILE="${out_dir}/.live_writer_stop"
  rm -f "${LIVE_WRITER_STOP_FILE}"
  python "${SCRIPT_DIR}/live_answer_writer.py" \
    --csv "${live_answers_csv}" \
    --jsonl "${live_answer_jsonl}" \
    --mode "${mode}" \
    --benchmark "${benchmark}" \
    --stop-file "${LIVE_WRITER_STOP_FILE}" &
  LIVE_WRITER_PID="$!"

  local -a cmd
  cmd=(
    lmms-eval
    eval
    --model "${EVAL_MODEL_BACKEND}"
    --model_args "${model_args}"
    --tasks "${benchmark}"
    --gen_kwargs "${gen_kwargs}"
    --batch_size "${BATCH_SIZE}"
    --output_path "${out_dir}"
  )

  if [[ "${LOG_SAMPLES:-1}" == "1" ]]; then
    cmd+=(--log_samples)
  fi

  if [[ -n "${LIMIT}" ]]; then
    cmd+=(--limit "${LIMIT}")
  fi

  cmd+=(--seed "${sample_seed}")

  log "RUN ${model_alias} | ${mode} | ${benchmark} | ${sweep_param}=${sweep_value} | seed=${sample_seed}"
  if env \
      LMMS_LIVE_ANSWERS_CSV="${live_answers_csv}" \
      LMMS_LIVE_MODE="${mode}" \
      LMMS_LIVE_MODEL_ALIAS="${model_alias}" \
      LMMS_MODEL_ALIAS="${model_alias}" \
      LMMS_MODEL_PATH="${model_path}" \
      LMMS_MAX_TOKENS="${run_max_tokens}" \
      LMMS_THINKING_BUDGET="${thinking_budget}" \
      LMMS_VLLM_VERSION="${VLLM_VERSION}" \
      LMMS_HARNESS_GIT_COMMIT="${HARNESS_GIT_COMMIT}" \
      LMMS_SAMPLE_SEED="${sample_seed}" \
      "${cmd[@]}" >"${run_log}" 2>&1; then
    stop_live_writer
    if [[ "${EXPORT_ARTIFACTS:-1}" == "1" ]]; then
      if ! export_run_artifacts \
          "${out_dir}" "${answer_model_name}" "${mode}" "${benchmark}" \
          "${sweep_param}" "${sweep_value}" \
          "${temperature}" "${top_p}" "${repetition_penalty}" "${presence_penalty}" "${sample_seed}" \
          "${model_alias}" "${model_path}" "${run_max_tokens}" "${thinking_budget}" >>"${run_log}" 2>&1; then
        cleanup_run_csvs "${out_dir}"
        touch "${marker_fail}"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        log "FAIL artifact export: ${model_alias} | ${mode} | ${benchmark} | ${sweep_param}=${sweep_value}"
        if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
          return 0
        fi
        return 1
      fi
    fi
    cleanup_run_csvs "${out_dir}"
    rm -f "${marker_fail}"
    touch "${marker_done}"
    TOTAL_OK=$((TOTAL_OK + 1))
    log "OK  ${model_alias} | ${mode} | ${benchmark} | ${sweep_param}=${sweep_value} | seed=${sample_seed}"
    return 0
  fi

  stop_live_writer
  cleanup_run_csvs "${out_dir}"
  touch "${marker_fail}"
  TOTAL_FAIL=$((TOTAL_FAIL + 1))
  log "FAIL ${model_alias} | ${mode} | ${benchmark} | ${sweep_param}=${sweep_value} | seed=${sample_seed} (see ${run_log})"

  if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
    return 0
  fi
  return 1
}

get_reason_baseline_temperature() {
  local model_alias="$1"
  echo "${REASON_BASELINE_TEMPERATURE_BY_MODEL[$model_alias]:-${REASON_BASELINE_TEMPERATURE}}"
}

get_reason_baseline_top_p() {
  local model_alias="$1"
  echo "${REASON_BASELINE_TOP_P_BY_MODEL[$model_alias]:-${REASON_BASELINE_TOP_P}}"
}

get_reason_baseline_repetition_penalty() {
  local model_alias="$1"
  echo "${REASON_BASELINE_REPETITION_PENALTY_BY_MODEL[$model_alias]:-${REASON_BASELINE_REPETITION_PENALTY}}"
}

get_reason_baseline_presence_penalty() {
  local model_alias="$1"
  echo "${REASON_BASELINE_PRESENCE_PENALTY_BY_MODEL[$model_alias]:-${REASON_BASELINE_PRESENCE_PENALTY}}"
}

run_sweeps_for_benchmark() {
  local model_alias="$1"
  local answer_model_name="$2"
  local mode="$3"
  local benchmark="$4"
  local model_path="$5"
  local baseline_temperature
  local baseline_top_p
  local baseline_repetition_penalty
  local baseline_presence_penalty

  if [[ "${mode}" == "non_reason" ]]; then
    baseline_temperature="${NON_REASON_BASELINE_TEMPERATURE:-${REASON_BASELINE_TEMPERATURE}}"
    baseline_top_p="${NON_REASON_BASELINE_TOP_P:-${REASON_BASELINE_TOP_P}}"
    baseline_repetition_penalty="${NON_REASON_BASELINE_REPETITION_PENALTY:-${REASON_BASELINE_REPETITION_PENALTY}}"
    baseline_presence_penalty="${NON_REASON_BASELINE_PRESENCE_PENALTY:-${REASON_BASELINE_PRESENCE_PENALTY}}"
  else
    baseline_temperature="$(get_reason_baseline_temperature "${model_alias}")"
    baseline_top_p="$(get_reason_baseline_top_p "${model_alias}")"
    baseline_repetition_penalty="$(get_reason_baseline_repetition_penalty "${model_alias}")"
    baseline_presence_penalty="$(get_reason_baseline_presence_penalty "${model_alias}")"
  fi

  log "BASELINE ${model_alias} | ${mode} | temp=${baseline_temperature} top_p=${baseline_top_p} repetition_penalty=${baseline_repetition_penalty} presence_penalty=${baseline_presence_penalty}"

  local seed
  local r
  for r in "${REPETITION_PENALTY_VALUES[@]}"; do
    for seed in "${SAMPLE_SEEDS[@]}"; do
      run_one_experiment \
        "${model_alias}" "${answer_model_name}" "${mode}" "${benchmark}" \
        "repetition_penalty" "${r}" \
        "${baseline_temperature}" "${baseline_top_p}" "${r}" "${baseline_presence_penalty}" "${seed}" "${model_path}" || return 1
    done
  done

  local p
  for p in "${PRESENCE_PENALTY_VALUES[@]}"; do
    for seed in "${SAMPLE_SEEDS[@]}"; do
      run_one_experiment \
        "${model_alias}" "${answer_model_name}" "${mode}" "${benchmark}" \
        "presence_penalty" "${p}" \
        "${baseline_temperature}" "${baseline_top_p}" "${baseline_repetition_penalty}" "${p}" "${seed}" "${model_path}" || return 1
    done
  done
}

run_mode() {
  local model_alias="$1"
  local answer_model_name="$2"
  local mode="$3"
  local model_path="$4"

  local -a benchmarks
  if [[ "${mode}" == "non_reason" ]]; then
    benchmarks=("${NON_REASON_BENCHMARKS[@]}")
  else
    benchmarks=("${REASON_BENCHMARKS[@]}")
  fi

  local benchmark
  for benchmark in "${benchmarks[@]}"; do
    run_sweeps_for_benchmark "${model_alias}" "${answer_model_name}" "${mode}" "${benchmark}" "${model_path}" || return 1
  done
}

model_is_selected() {
  local alias="$1"
  if [[ ${#SELECTED_MODEL_ALIASES[@]} -eq 0 ]]; then
    return 0
  fi
  local selected
  for selected in "${SELECTED_MODEL_ALIASES[@]}"; do
    if [[ "${alias}" == "${selected}" ]]; then
      return 0
    fi
  done
  return 1
}

main() {
  log "Results root: ${RUN_ROOT}"

  local i
  for i in "${!MODEL_PATHS[@]}"; do
    local model_path="${MODEL_PATHS[$i]}"
    local model_alias="${MODEL_ALIASES[$i]}"
    local answer_model_name="${ANSWER_MODEL_NAMES[$i]}"

    if ! model_is_selected "${model_alias}"; then
      log "SKIP unselected model: ${model_alias}"
      continue
    fi

    local model_root="${RUN_ROOT}/${model_alias}"
    mkdir -p "${model_root}"

    local server_log="${model_root}/vllm_server.log"
    if ! start_server "${model_path}" "${model_alias}" "${server_log}"; then
      TOTAL_FAIL=$((TOTAL_FAIL + 1))
      if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
        exit 1
      fi
      continue
    fi

    if [[ "${RUN_NON_REASON:-1}" == "1" ]]; then
      run_mode "${model_alias}" "${answer_model_name}" "non_reason" "${model_path}" || {
        if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
          exit 1
        fi
      }
    else
      log "SKIP non_reason for ${model_alias} (RUN_NON_REASON=${RUN_NON_REASON:-1})"
    fi

    if [[ "${RUN_REASON:-1}" == "1" ]]; then
      run_mode "${model_alias}" "${answer_model_name}" "reason" "${model_path}" || {
        if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
          exit 1
        fi
      }
    else
      log "SKIP reason for ${model_alias} (RUN_REASON=${RUN_REASON:-1})"
    fi

    cleanup
    SERVER_PID=""
  done

  log "Completed. Success=${TOTAL_OK}, Failed=${TOTAL_FAIL}"
  if [[ "${TOTAL_FAIL}" -gt 0 ]]; then
    exit 2
  fi
}

main "$@"
