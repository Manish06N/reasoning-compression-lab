#!/usr/bin/env bash
# A–Z preflight for Experiment A (qrm-official). Exit 0 = ready; non-zero = fix before submit.
set -eo pipefail

export PATH="/usr/bin:/bin:${PATH}"

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
QRM_DIR="${QRM_REPO_DIR:-$QR/external/Quantized-Reasoning-Models}"
ENV_NAME="${QRM_CONDA_ENV:-qrm-official}"
CONDA_ROOT="${CONDA_ROOT:-/home/apps/MSCC/miniconda3}"
MARKER="$QR/.qrm_official_env_ready"
INSTALL_REV="2026-07-06-conda-gcc12-nvcc124-vllm070wheel"
MODEL="${QRM_MODEL_PATH:-$QR/models/DeepSeek-R1-Distill-Qwen-7B}"
VLLM_SO_MIN_MB=180
VLLM_SO_MAX_MB=280
FAIL=0

pass() { echo "  [PASS] $*"; }
fail() { echo "  [FAIL] $*" >&2; FAIL=1; }

echo "=== QRM official preflight (A–Z) ==="
echo "QR=$QR"
echo ""

echo "--- A. Repo layout ---"
[[ -d "$QR" ]] && pass "project root" || fail "missing $QR"
[[ -d "$QRM_DIR/.git" ]] && pass "QRM clone" || fail "missing $QRM_DIR — run setup_official_qrm_repo.sh"
for sub in third-party/lighteval third-party/vllm third-party/fast-hadamard-transform; do
  if [[ -d "$QRM_DIR/$sub" ]]; then
    pass "submodule $sub"
  else
    fail "missing submodule $sub"
  fi
done
[[ -f "$QRM_DIR/inference.py" ]] && pass "inference.py" || fail "missing inference.py"
[[ -x "$QR/scripts/hpc/qrm_parity/install_official_qrm_env.sh" ]] && pass "install script" || fail "install script"
[[ -x "$QR/scripts/hpc/qrm_parity/run_official_inference.sh" ]] && pass "inference runner" || fail "inference runner"
grep -q '#SBATCH --gres=gpu:1' "$QR/slurm/qrm_official_math500_n10.slurm" && pass "slurm gres/gpu:1" || fail "slurm missing gres/gpu:1"
! grep -q '#SBATCH --exclusive' "$QR/slurm/qrm_official_math500_n10.slurm" && pass "slurm non-exclusive" || fail "slurm should not be --exclusive"

echo ""
echo "--- B. Env marker ---"
if [[ -f "$MARKER" ]] && grep -qxF "$INSTALL_REV" "$MARKER"; then
  pass "marker $INSTALL_REV"
else
  fail "marker missing or stale (have: $(cat "$MARKER" 2>/dev/null || echo none); want: $INSTALL_REV)"
fi

echo ""
echo "--- C. Model & dataset ---"
if [[ -d "$MODEL" && -f "$MODEL/config.json" ]]; then
  pass "model $MODEL"
else
  fail "model missing or incomplete: $MODEL"
fi
DATA_DIR="$QRM_DIR/datasets/MATH-500"
if [[ -d "$DATA_DIR" ]] && [[ -n "$(ls -A "$DATA_DIR" 2>/dev/null)" ]]; then
  pass "MATH-500 dataset at $DATA_DIR"
else
  fail "MATH-500 dataset missing — run prepare_qrm_datasets.sh"
fi

echo ""
echo "--- D. Conda env & toolchain ---"
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export PATH="$CONDA_PREFIX/bin:/usr/bin:/bin:$PATH"
if [[ -x "$CONDA_PREFIX/bin/git" ]]; then
  export GIT_PYTHON_GIT_EXECUTABLE="$CONDA_PREFIX/bin/git"
elif command -v git >/dev/null 2>&1; then
  export GIT_PYTHON_GIT_EXECUTABLE="$(command -v git)"
fi

[[ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" ]] && pass "conda gcc" || fail "conda gcc missing"
[[ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" ]] && pass "conda g++" || fail "conda g++ missing"
command -v nvcc >/dev/null 2>&1 && pass "nvcc $(nvcc --version | head -1)" || fail "nvcc missing"
[[ -x "$CONDA_PREFIX/bin/git" ]] && pass "conda git" || fail "conda git missing"

gcc_ver="$("$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" -dumpversion 2>/dev/null | cut -d. -f1)"
if [[ "${gcc_ver:-99}" -le 13 ]]; then
  pass "gcc major version $gcc_ver (nvcc-compatible)"
else
  fail "gcc $gcc_ver too new for nvcc 12.4 (need <=13)"
fi

echo ""
echo "--- E. Python packages & binaries ---"
cd "$QRM_DIR"
export PYTHONPATH="$QRM_DIR${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY' || { echo "  [FAIL] Python import check"; FAIL=1; }
import importlib
import os
import sys

checks = []

def ok(msg):
    checks.append(("PASS", msg))
    print(f"  [PASS] {msg}")

def bad(msg):
    checks.append(("FAIL", msg))
    print(f"  [FAIL] {msg}", file=sys.stderr)

import torch
ok(f"torch {torch.__version__}")
if not torch.__version__.startswith("2.5"):
    bad(f"torch version unexpected: {torch.__version__}")

import transformers
ok(f"transformers {transformers.__version__}")

import fast_hadamard_transform
ok("fast_hadamard_transform import")

import lighteval
ok(f"lighteval {getattr(lighteval, '__version__', 'unknown')}")

import git
exe = os.environ.get("GIT_PYTHON_GIT_EXECUTABLE") or git.Git().GIT_PYTHON_GIT_EXECUTABLE
if exe and os.path.isfile(exe) and os.access(exe, os.X_OK):
    ok(f"GitPython executable {exe}")
else:
    bad(f"GitPython bad executable: {exe!r}")

import vllm
ver = getattr(vllm, "__version__", "unknown")
ok(f"vllm {ver}")
if not ver.startswith("0.7"):
    bad(f"vllm version unexpected: {ver}")

from vllm_custom.model_executor.fake_quantized_models.registry import register_fake_quantized_models
register_fake_quantized_models()
ok("vllm_custom registry")

# lighteval path used at inference time
from lighteval.logging.evaluation_tracker import EvaluationTracker  # noqa: F401
ok("lighteval EvaluationTracker (needs git)")

failed = [m for s, m in checks if s == "FAIL"]
sys.exit(1 if failed else 0)
PY

echo ""
echo "--- F. Install script dry-run (should skip if marker ok) ---"

VLLM_SO="$QRM_DIR/third-party/vllm/vllm/_C.abi3.so"
if [[ -f "$VLLM_SO" ]]; then
  so_mb=$(( $(stat -c%s "$VLLM_SO") / 1024 / 1024 ))
  if [[ "$so_mb" -ge "$VLLM_SO_MIN_MB" && "$so_mb" -le "$VLLM_SO_MAX_MB" ]]; then
    pass "vllm _C.abi3.so size ${so_mb}MB (expected ~215MB PyPI 0.7.0)"
  else
    fail "vllm _C.abi3.so size ${so_mb}MB — wrong wheel? (want ${VLLM_SO_MIN_MB}-${VLLM_SO_MAX_MB}MB)"
  fi
else
  fail "missing $VLLM_SO"
fi

if bash "$QR/scripts/hpc/qrm_parity/install_official_qrm_env.sh" 2>&1 | grep -q "Skipping pip install"; then
  pass "install_official_qrm_env.sh skips correctly"
else
  fail "install script did not skip — marker mismatch?"
fi

echo ""
if [[ "$FAIL" -eq 0 ]]; then
  echo "=== PREFLIGHT OK — safe to submit Experiment A ==="
  echo "Note: vllm._C CUDA load is verified only on GPU node (login has no libcuda)."
  exit 0
else
  echo "=== PREFLIGHT FAILED — fix items above before submitting ==="
  exit 1
fi