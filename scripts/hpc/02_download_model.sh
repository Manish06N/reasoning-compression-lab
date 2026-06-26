#!/usr/bin/env bash
set -euo pipefail

export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qreason

MODEL_DIR="${QR}/models/DeepSeek-R1-Distill-Qwen-7B"
mkdir -p "$MODEL_DIR"

echo "Downloading DeepSeek-R1-Distill-Qwen-7B to $MODEL_DIR"
echo "This can take 20-60 minutes depending on cluster network speed."

huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local-dir "$MODEL_DIR"

echo "Download complete."
echo "Set model path with:"
echo "  export QREASON_MODEL_QWEN7B=$MODEL_DIR"
