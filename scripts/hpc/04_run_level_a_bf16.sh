#!/usr/bin/env bash
set -euo pipefail

export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qreason

export QREASON_MODEL_QWEN7B="${QREASON_MODEL_QWEN7B:-$QR/models/DeepSeek-R1-Distill-Qwen-7B}"

LIMIT="${1:-}"
if [[ -n "$LIMIT" ]]; then
  python scripts/run_inference.py \
    --cell-config configs/cells/level_a_bf16_seed0.json \
    --limit "$LIMIT"
else
  python scripts/run_inference.py \
    --cell-config configs/cells/level_a_bf16_seed0.json
fi

echo "Raw output: runs/raw/level_a_qwen7b_bf16_math500_seed0.jsonl"
