#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=param_rudra_env.sh
source "${SCRIPT_DIR}/param_rudra_env.sh"

cd "$QR"
param_rudra_activate_conda

python scripts/score_run.py \
  --input runs/raw/level_a_qwen7b_bf16_math500_seed0.jsonl

echo "Scored output: runs/scored/level_a_qwen7b_bf16_math500_seed0.jsonl"
echo "Summary: results/level_a_qwen7b_bf16_math500_seed0_summary.json"
