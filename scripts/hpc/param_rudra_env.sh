#!/usr/bin/env bash
# PARAM Rudra (IIT Patna) environment — sourced by scripts/hpc/*.sh on the cluster.

export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"

# Keep Hugging Face cache on scratch (not home).
export HF_HOME="${HF_HOME:-${QR}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

# Model path default for Level A.
export QREASON_MODEL_QWEN7B="${QREASON_MODEL_QWEN7B:-${QR}/models/DeepSeek-R1-Distill-Qwen-7B}"

param_rudra_load_modules() {
  if ! command -v module >/dev/null 2>&1; then
    return 0
  fi
  module load mldl/Miniconda 2>/dev/null || true
}

param_rudra_activate_conda() {
  param_rudra_load_modules
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found. Run: module load mldl/Miniconda"
    return 1
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate qreason
}
