#!/bin/bash
#SBATCH --job-name=qreason-hpc-b01
#SBATCH --output=logs/slurm/b01_parallel_bf16_%j.out
#SBATCH --error=logs/slurm/b01_parallel_bf16_%j.err
#SBATCH --time=47:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2

set -euo pipefail
export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"
mkdir -p logs/slurm

bash scripts/hpc/run_hpc_2a100_publication.sh b01_parallel_bf16_anchors
