#!/usr/bin/env bash
set -euo pipefail

export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qreason

export QREASON_MODEL_QWEN7B="${QREASON_MODEL_QWEN7B:-$QR/models/DeepSeek-R1-Distill-Qwen-7B}"

python scripts/smoke_test.py \
  --cell-config configs/cells/level_a_bf16_seed0.json \
  --decoding-config configs/decoding/smoke_debug.yaml \
  --limit 3 \
  --output runs/raw/smoke_test.jsonl

echo "Smoke test output: runs/raw/smoke_test.jsonl"
