#!/bin/bash
#SBATCH --job-name=qreason-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/smoke_test_%j.out
#SBATCH --error=logs/smoke_test_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8

set -euo pipefail
export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"
mkdir -p logs runs/raw

bash scripts/hpc/03_smoke_test.sh
