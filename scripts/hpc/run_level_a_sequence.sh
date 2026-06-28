#!/usr/bin/env bash
# Run on HPC after exclusive smoke passes (Gate 3 → 4).
# Usage: bash scripts/hpc/run_level_a_sequence.sh [debug_limit]
set -euo pipefail

export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"

DEBUG_LIMIT="${1:-10}"

# shellcheck disable=SC1091
source "$QR/scripts/hpc/param_rudra_env.sh" 2>/dev/null || true
param_rudra_activate_conda 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate qreason

export QREASON_MODEL_QWEN7B="${QREASON_MODEL_QWEN7B:-$QR/models/DeepSeek-R1-Distill-Qwen-7B}"

echo "=== Gate 3: smoke test ==="
bash scripts/hpc/03_smoke_test.sh

echo ""
echo "=== Gate 4a: BF16 debug (limit=$DEBUG_LIMIT) ==="
bash scripts/hpc/04_run_level_a_bf16.sh "$DEBUG_LIMIT"

echo ""
echo "=== Gate 4b: BF16 full MATH-500 ==="
bash scripts/hpc/04_run_level_a_bf16.sh 500

echo ""
echo "=== Score Level A ==="
bash scripts/hpc/05_score_level_a.sh

echo ""
echo "Check: results/level_a_qwen7b_bf16_math500_seed0_summary.json"
