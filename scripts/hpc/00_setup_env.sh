#!/usr/bin/env bash
# Run on HPC login node (or inside an interactive GPU session after modules load).
set -euo pipefail

export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"

export HF_HOME="${HF_HOME:-$QR/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

echo "Workspace: $QR"

CONDA_ROOT="${CONDA_ROOT:-/home/apps/MSCC/miniconda3}"

if [[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  # PARAM Rudra provides conda here; the module system is not used for Python.
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
elif command -v module >/dev/null 2>&1; then
  module avail cuda 2>&1 | head -20 || true
  echo "Edit this script to uncomment the correct CUDA module for your cluster."
  # module load cuda/12.1
  # module load anaconda
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Set CONDA_ROOT or load your cluster conda."
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx qreason; then
  echo "Conda env qreason already exists."
else
  conda create -n qreason python=3.11 -y
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qreason

pip install --upgrade pip
pip install -r requirements-hpc.txt

echo "Environment ready. Test with: bash scripts/hpc/01_gpu_check.sh"
