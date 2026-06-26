#!/bin/bash
#SBATCH --job-name=qreason-download
#SBATCH --partition=cpu
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"
mkdir -p logs models

module load mldl/Miniconda 2>/dev/null || true

bash scripts/hpc/02_download_model.sh
