#!/usr/bin/env bash
# Submit Phase 0 3-question smoke tests for Qwen-7B and Llama-8B (BF16 & FP8)
set -euo pipefail

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"
mkdir -p logs

SMOKE_ROOT="${QRM_OUTPUT_ROOT:-$QR/outputs-hpc-phase0-smoke-$(date +%Y-%m-%d)}"
mkdir -p "$SMOKE_ROOT"

QWEN_BF16="$QR/models/DeepSeek-R1-Distill-Qwen-7B"
LLAMA_BF16="$QR/models/DeepSeek-R1-Distill-Llama-8B"

echo "=== Submitting Phase 0 Smoke Tests (n=3, seed=42) ==="

JOB_QWEN_BF16=$(
  QRM_MODEL_PATH="$QWEN_BF16" \
  QRM_OUTPUT_ROOT="$SMOKE_ROOT" \
  QRM_MAX_SAMPLES=3 \
  QRM_SEED=42 \
  sbatch --parsable --job-name=qreason-smoke-qwen-bf16 \
    slurm/qrm_official_math500_n10.slurm
)

JOB_LLAMA_BF16=$(
  QRM_MODEL_PATH="$LLAMA_BF16" \
  QRM_OUTPUT_ROOT="$SMOKE_ROOT" \
  QRM_MAX_SAMPLES=3 \
  QRM_SEED=42 \
  sbatch --parsable --job-name=qreason-smoke-llama-bf16 \
    slurm/qrm_official_math500_n10.slurm
)

printf "Submitted Phase 0 BF16 Smoke Tests:\n"
printf "  Qwen-7B BF16 Job:  %s\n" "$JOB_QWEN_BF16"
printf "  Llama-8B BF16 Job: %s\n" "$JOB_LLAMA_BF16"
printf "  Output Root:       %s\n" "$SMOKE_ROOT"
