#!/usr/bin/env bash
# Install official QRM evaluation stack in a dedicated conda env (separate from qreason).
set -euo pipefail

# Compute nodes may hide system git/gcc after conda activate (see KNOWN_ISSUES §3b).
export PATH="/usr/bin:/bin:${PATH}"

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
QRM_DIR="${QRM_REPO_DIR:-$QR/external/Quantized-Reasoning-Models}"
ENV_NAME="${QRM_CONDA_ENV:-qrm-official}"
CONDA_ROOT="${CONDA_ROOT:-/home/apps/MSCC/miniconda3}"
MARKER="$QR/.qrm_official_env_ready"
INSTALL_REV="2026-07-06-conda-nvcc-vllm07"

if [[ ! -d "$QRM_DIR/.git" ]]; then
  bash "$QR/scripts/hpc/qrm_parity/setup_official_qrm_repo.sh"
fi

cd "$QRM_DIR"
HAD_DIR="$QRM_DIR/third-party/fast-hadamard-transform"
if [[ ! -f "$HAD_DIR/setup.py" && ! -f "$HAD_DIR/pyproject.toml" ]]; then
  if command -v git >/dev/null 2>&1; then
    echo "=== Initializing QRM submodules (lighteval, vllm, fast-hadamard-transform) ==="
    git submodule update --init --recursive \
      third-party/lighteval \
      third-party/vllm \
      third-party/fast-hadamard-transform
  else
    echo "ERROR: fast-hadamard-transform submodule missing and git unavailable on this node." >&2
    echo "  Run on login node: bash scripts/hpc/qrm_parity/setup_official_qrm_repo.sh" >&2
    exit 1
  fi
else
  echo "=== QRM submodules present (init skipped) ==="
fi

if [[ -f "$MARKER" ]] && grep -qxF "$INSTALL_REV" "$MARKER"; then
  echo "QRM official env already installed ($MARKER: $INSTALL_REV). Skipping pip install."
  exit 0
fi

if [[ -f "$MARKER" ]]; then
  echo "WARN: stale QRM env marker ($(cat "$MARKER")); reinstalling stack." >&2
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env $ENV_NAME exists"
else
  conda create -n "$ENV_NAME" python=3.11 -y
fi
conda activate "$ENV_NAME"
export PATH="/usr/bin:/bin:${PATH}"

pip install -U pip wheel setuptools

echo "=== Installing QRM base requirements (pins torch 2.5.1; overwrites bad pip vllm fallback) ==="
pip install -r "$QRM_DIR/requirements.txt"
# Remove packages left by the failed pip vllm==0.8.5 fallback (job 87130).
pip uninstall -y vllm torchvision torchaudio xformers 2>/dev/null || true

echo "=== Ensuring CUDA compiler (nvcc) — HPC GPU nodes have drivers only, no system toolkit ==="
if ! command -v nvcc >/dev/null 2>&1; then
  conda install -y -c nvidia cuda-nvcc=12.4 cuda-cudart-dev=12.4
fi
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export PATH="$CONDA_PREFIX/bin:$PATH"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc still missing after conda cuda-nvcc install (CUDA_HOME=$CUDA_HOME)" >&2
  exit 1
fi
echo "Using nvcc: $(command -v nvcc)"

echo "=== Installing fast-hadamard-transform (QRM submodule) ==="
pip install --no-build-isolation -e "$QRM_DIR/third-party/fast-hadamard-transform"

echo "=== Installing lighteval (QRM fork) ==="
pip install -e "$QRM_DIR/third-party/lighteval[math]"

echo "=== Installing QRM vLLM fork v0.7.0 (precompiled wheels) ==="
export VLLM_USE_PRECOMPILED=1
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.7.0
pip install -e "$QRM_DIR/third-party/vllm" || {
  echo "WARN: editable vllm failed; retrying with --no-build-isolation" >&2
  pip install --no-build-isolation -e "$QRM_DIR/third-party/vllm"
}

pip install typer rich

echo "=== Verifying QRM stack imports ==="
python - <<'PY'
import fast_hadamard_transform  # noqa: F401
from vllm_custom.model_executor.fake_quantized_models.registry import register_fake_quantized_models
register_fake_quantized_models()
import vllm
print(f"fast_hadamard_transform: ok")
print(f"vllm_custom registry: ok")
print(f"vllm version: {getattr(vllm, '__version__', 'unknown')}")
PY

printf '%s\n' "$INSTALL_REV" >"$MARKER"
echo "QRM official env ready: conda activate $ENV_NAME"