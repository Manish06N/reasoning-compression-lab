#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=param_rudra_env.sh
source "${SCRIPT_DIR}/param_rudra_env.sh"

cd "$QR"
param_rudra_activate_conda

MODEL_DIR="${QREASON_MODEL_QWEN7B}"
mkdir -p "$MODEL_DIR"

echo "Downloading DeepSeek-R1-Distill-Qwen-7B to $MODEL_DIR"
echo "This can take 20-60 minutes depending on cluster network speed."

huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local-dir "$MODEL_DIR"

echo "Download complete."
echo "Set model path with:"
echo "  export QREASON_MODEL_QWEN7B=$MODEL_DIR"
