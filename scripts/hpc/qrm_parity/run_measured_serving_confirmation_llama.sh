#!/bin/bash
#SBATCH --job-name=srv_llama
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=47:00:00
#SBATCH --output=logs/measured_serving_confirmation_llama_%j.out
#SBATCH --error=logs/measured_serving_confirmation_llama_%j.err

# ==============================================================================
# MEASURED SERVING CONFIRMATION BENCHMARK RUNNER (LLAMA-8B CHANNEL)
# Author: Manish Nandish, IIT Patna
# Cluster: PARAM Rudra HPC (NVIDIA A100-PCIE-80GB)
# ==============================================================================

set -eo pipefail

REPO_ROOT="/scratch/manishn_iitp/reasoning-compression-lab"
cd "${REPO_ROOT}"
mkdir -p logs results/measured_serving_confirmation/raw results/reports/measured_serving_confirmation/provenance

echo "=============================================================================="
echo "STARTING LLAMA-8B CONFIRMATION BENCHMARK ON PARAM RUDRA HPC"
echo "Host: $(hostname) | SLURM Job ID: ${SLURM_JOB_ID:-N/A} | Date: $(date)"
echo "=============================================================================="

export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=INFO
export PATH="/home/manishn_iitp/.conda/envs/qrm-official/bin:${PATH}"

PYTHON_BIN="/home/manishn_iitp/.conda/envs/qrm-official/bin/python3"

MODELS_DIR="${REPO_ROOT}/models"
LLAMA_BF16="${MODELS_DIR}/DeepSeek-R1-Distill-Llama-8B"
LLAMA_FP8="${MODELS_DIR}/DeepSeek-R1-Distill-Llama-8B-FP8"
LLAMA_AWQ="${MODELS_DIR}/DeepSeek-R1-Distill-Llama-8B-AWQ-4"
LLAMA_GPTQ="${MODELS_DIR}/DeepSeek-R1-Distill-Llama-8B-GPTQ-4"

# Run all 4 Llama-8B formats sequentially on this physical A100 GPU
echo ">>> Running Llama-8B BF16..."
"${PYTHON_BIN}" scripts/hpc/qrm_parity/benchmark_serving_confirmation.py --model-path "${LLAMA_BF16}" --model-name "Llama-8B" --format "BF16"

echo ">>> Running Llama-8B FP8..."
"${PYTHON_BIN}" scripts/hpc/qrm_parity/benchmark_serving_confirmation.py --model-path "${LLAMA_FP8}" --model-name "Llama-8B" --format "FP8"

echo ">>> Running Llama-8B AWQ-4..."
"${PYTHON_BIN}" scripts/hpc/qrm_parity/benchmark_serving_confirmation.py --model-path "${LLAMA_AWQ}" --model-name "Llama-8B" --format "AWQ-4"

echo ">>> Running Llama-8B GPTQ-4..."
"${PYTHON_BIN}" scripts/hpc/qrm_parity/benchmark_serving_confirmation.py --model-path "${LLAMA_GPTQ}" --model-name "Llama-8B" --format "GPTQ-4"

echo "=============================================================================="
echo "LLAMA-8B BENCHMARK RUNS COMPLETED AT $(date)"
echo "=============================================================================="
