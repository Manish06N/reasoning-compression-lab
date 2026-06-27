#!/bin/bash
#SBATCH --job-name=qreason-smoke-quick
#SBATCH --output=logs/smoke_quick_%j.out
#SBATCH --error=logs/smoke_quick_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclusive

set -euo pipefail
export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"
mkdir -p logs

export SMOKE_LIMIT="${SMOKE_LIMIT:-1}"
export SMOKE_MAX_TOKENS="${SMOKE_MAX_TOKENS:-64}"
export SMOKE_OUTPUT="${SMOKE_OUTPUT:-runs/raw/smoke_test_quick.jsonl}"

bash scripts/hpc/03_smoke_test.sh
