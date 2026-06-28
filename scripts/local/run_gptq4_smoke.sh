#!/usr/bin/env bash
# GPTQ-4 smoke after download + verify (Phase 2 local).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

bash "$QR/scripts/hpc/06_verify_gptq4_model.sh"

python scripts/smoke_test.py \
  --cell-config configs/cells/level_a_gptq4_seed0.json \
  --limit 1 \
  --max-tokens 64 \
  --output runs/raw/smoke_qwen7b_gptq4_local.jsonl

echo "GPTQ-4 smoke complete: runs/raw/smoke_qwen7b_gptq4_local.jsonl"
