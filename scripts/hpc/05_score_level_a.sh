#!/usr/bin/env bash
set -euo pipefail

export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qreason

python scripts/score_run.py \
  --input runs/raw/level_a_qwen7b_bf16_math500_seed0.jsonl

echo "Scored output: runs/scored/level_a_qwen7b_bf16_math500_seed0.jsonl"
echo "Summary: results/level_a_qwen7b_bf16_math500_seed0_summary.json"
