#!/bin/bash
#SBATCH --job-name=qreason-hpc-b02-gpqa
#SBATCH --output=logs/slurm/b02_gpqa_%j.out
#SBATCH --error=logs/slurm/b02_gpqa_%j.err
#SBATCH --time=47:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -euo pipefail
export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"
mkdir -p logs/slurm

bash scripts/hpc/run_hpc_2a100_publication.sh b02_gpqa_fp8
