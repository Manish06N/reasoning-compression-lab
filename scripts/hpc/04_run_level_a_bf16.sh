#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=param_rudra_env.sh
source "${SCRIPT_DIR}/param_rudra_env.sh"

cd "$QR"
param_rudra_activate_conda

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
