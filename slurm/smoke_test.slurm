#!/bin/bash
#SBATCH --job-name=qreason-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/smoke_test_%j.out
#SBATCH --error=logs/smoke_test_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

set -euo pipefail
export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"
mkdir -p logs runs/raw

module load mldl/Miniconda 2>/dev/null || true

bash scripts/hpc/03_smoke_test.sh
