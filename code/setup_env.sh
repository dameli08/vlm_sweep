#!/usr/bin/env bash
set -euo pipefail

# Fresh environment installer for VLM sweep automation.
# This script does not touch any existing project files.

ENV_NAME="${1:-vlm_sweep_20260617}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Conda env ${ENV_NAME} already exists."
else
  echo "[INFO] Creating conda env ${ENV_NAME} (python=${PYTHON_VERSION})"
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

echo "[INFO] Installing core packages..."
python -m pip install --upgrade pip wheel setuptools

# Install vLLM and lmms-eval in this new env.
python -m pip install "vllm"
python -m pip install "git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git"

# lmms-eval currently installs both legacy latex2sympy2 and
# latex2sympy2-extended through math-verify. Their ANTLR metadata conflicts,
# but 4.9.3 is the oldest runtime that successfully parses with both.
python -m pip install --upgrade --no-deps "antlr4-python3-runtime==4.9.3"
python "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patch_lmms_reasoning_capture.py"

# Repair known packaging issue in some lmms-eval builds where include files
# referenced by task YAMLs may be missing from the wheel/source package.
python - <<'PY'
from pathlib import Path
import importlib.util
import re

spec = importlib.util.find_spec("lmms_eval")
if spec is None or spec.origin is None:
  raise SystemExit("lmms_eval is not importable after install")

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
for p in sorted(unresolved)[:20]:
  print(f"[WARN] unresolved include target: {p}")
PY

echo "[INFO] Verifying install..."
python - <<'PY'
import importlib.util as iu
mods = ["vllm", "lmms_eval", "latex2sympy2", "latex2sympy2_extended", "math_verify"]
missing = [m for m in mods if iu.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing modules: {missing}")
print("All required modules are installed:", mods)

from latex2sympy2 import latex2sympy as legacy_latex2sympy
from latex2sympy2_extended.latex2sympy2 import latex2sympy as extended_latex2sympy
from math_verify import parse as math_verify_parse

legacy_latex2sympy(r"x^2")
extended_latex2sympy(r"x^2")
math_verify_parse(r"$x^2$")
print("Both LaTeX parser paths passed a smoke test.")
PY

if command -v lmms-eval >/dev/null 2>&1; then
  echo "[INFO] lmms-eval CLI is available."
else
  echo "[WARN] lmms-eval CLI not found in PATH; try reopening terminal after activation."
fi

echo "[DONE] Environment is ready. Activate with: conda activate ${ENV_NAME}"
