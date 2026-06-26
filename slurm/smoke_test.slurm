#!/bin/bash
#SBATCH --job-name=qreason-smoke
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1

# EDIT FOR YOUR CLUSTER:
# #SBATCH --partition=gpu
# #SBATCH --account=your_account

set -euo pipefail
export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"
mkdir -p logs

# module load cuda/12.1
# module load anaconda

bash scripts/hpc/03_smoke_test.sh
