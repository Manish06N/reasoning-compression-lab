#!/bin/bash
#SBATCH --job-name=qreason-level-a-bf16
#SBATCH --output=logs/level_a_bf16_%j.out
#SBATCH --error=logs/level_a_bf16_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -euo pipefail
export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"
mkdir -p logs

# Optional debug first: pass a small limit via env, e.g. LIMIT=10 sbatch ...
if [[ -n "${LIMIT:-}" ]]; then
  bash scripts/hpc/04_run_level_a_bf16.sh "$LIMIT"
else
  bash scripts/hpc/04_run_level_a_bf16.sh
fi

bash scripts/hpc/05_score_level_a.sh
