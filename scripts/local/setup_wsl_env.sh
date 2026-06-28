#!/usr/bin/env bash
# One-time WSL setup for Windows RTX 5080 local runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$QR"

source ~/miniconda3/etc/profile.d/conda.sh

if ! conda env list | awk '{print $1}' | grep -qx qreason; then
  echo "Creating conda env qreason (python 3.11)..."
  conda create -n qreason python=3.11 -y
fi

conda activate qreason
pip install --upgrade pip
pip install -r requirements-hpc.txt

export HF_HOME="${HF_HOME:-$QR/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$QR/models" "$QR/runs/raw" "$QR/results"

echo ""
echo "Setup complete."
echo "  Repo: $QR"
echo "  Env:  conda activate qreason"
echo "  Next: bash scripts/local/gpu_check.sh"
