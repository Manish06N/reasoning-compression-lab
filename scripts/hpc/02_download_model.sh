#!/usr/bin/env bash
set -euo pipefail

export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"

export HF_HOME="${HF_HOME:-$QR/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

CONDA_ROOT="${CONDA_ROOT:-/home/apps/MSCC/miniconda3}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate qreason

MODEL_DIR="${QR}/models/DeepSeek-R1-Distill-Qwen-7B"
mkdir -p "$MODEL_DIR"

echo "Downloading DeepSeek-R1-Distill-Qwen-7B to $MODEL_DIR"
echo "This can take 20-60 minutes depending on cluster network speed."

hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local-dir "$MODEL_DIR"

echo "Download complete."
echo "Set model path with:"
echo "  export QREASON_MODEL_QWEN7B=$MODEL_DIR"
