#!/bin/bash
#SBATCH --job-name=qreason-qrm_official_n10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/reasoning-compression-lab/logs/qrm_official_%j.out
#SBATCH --error=/scratch/%u/reasoning-compression-lab/logs/qrm_official_%j.err

set -eo pipefail

QR="/scratch/${USER}/reasoning-compression-lab"
cd "$QR"
mkdir -p logs

export QR
export QRM_MAX_SAMPLES="${QRM_MAX_SAMPLES:-10}"
export QRM_SEED="${QRM_SEED:-42}"
export QRM_MODEL_PATH="${QRM_MODEL_PATH:-$QR/models/DeepSeek-R1-Distill-Qwen-7B}"
export QRM_OUTPUT_ROOT="${QRM_OUTPUT_ROOT:-$QR/outputs-hpc-qrm-official-$(date +%Y-%m-%d)}"
export HF_HOME="${HF_HOME:-$QR/hf_cache}"

echo "=== QRM official stack install (first run may take 30-60 min) ==="
bash scripts/hpc/qrm_parity/install_official_qrm_env.sh

echo "=== QRM official inference.py MATH-500 n=$QRM_MAX_SAMPLES seed=$QRM_SEED ==="
bash scripts/hpc/qrm_parity/run_official_inference.sh

echo "=== Done ==="