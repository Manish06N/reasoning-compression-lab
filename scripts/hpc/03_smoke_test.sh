#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=param_rudra_env.sh
source "${SCRIPT_DIR}/param_rudra_env.sh"

cd "$QR"
param_rudra_activate_conda

python scripts/smoke_test.py \
  --cell-config configs/cells/level_a_bf16_seed0.json \
  --decoding-config configs/decoding/smoke_debug.yaml \
  --limit 3 \
  --output runs/raw/smoke_test.jsonl

echo "Smoke test output: runs/raw/smoke_test.jsonl"
