#!/usr/bin/env bash
# Install official QRM evaluation stack in a dedicated conda env (separate from qreason).
set -euo pipefail

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
QRM_DIR="${QRM_REPO_DIR:-$QR/external/Quantized-Reasoning-Models}"
ENV_NAME="${QRM_CONDA_ENV:-qrm-official}"
CONDA_ROOT="${CONDA_ROOT:-/home/apps/MSCC/miniconda3}"
MARKER="$QR/.qrm_official_env_ready"

if [[ ! -d "$QRM_DIR/.git" ]]; then
  bash "$QR/scripts/hpc/qrm_parity/setup_official_qrm_repo.sh"
fi

echo "=== Initializing QRM submodules (lighteval + vllm) ==="
cd "$QRM_DIR"
git submodule update --init --recursive third-party/lighteval third-party/vllm

if [[ -f "$MARKER" ]]; then
  echo "QRM official env already installed ($MARKER). Skipping pip install."
  exit 0
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env $ENV_NAME exists"
else
  conda create -n "$ENV_NAME" python=3.11 -y
fi
conda activate "$ENV_NAME"

pip install -U pip wheel setuptools
pip install -r "$QRM_DIR/requirements.txt"

echo "=== Installing lighteval (QRM fork) ==="
pip install -e "$QRM_DIR/third-party/lighteval[math]"

echo "=== Installing QRM vLLM fork (precompiled wheels) ==="
export VLLM_USE_PRECOMPILED=1
pip install -e "$QRM_DIR/third-party/vllm" || {
  echo "WARN: QRM vllm editable install failed; falling back to pip vllm==0.8.5" >&2
  pip install "vllm==0.8.5"
}

pip install typer rich

touch "$MARKER"
echo "QRM official env ready: conda activate $ENV_NAME"