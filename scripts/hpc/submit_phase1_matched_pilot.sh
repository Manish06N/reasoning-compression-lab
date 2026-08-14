#!/usr/bin/env bash
# Submit Phase 1 Matched Pilot Runs (MATH-500 n=500, seed 42)
# Compares BF16 vs FP8 across Qwen-7B and Llama-8B under identical protocol
set -euo pipefail

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"
mkdir -p logs

OUTPUT_ROOT="${QRM_OUTPUT_ROOT:-$QR/outputs-hpc-phase1-pilot-$(date +%Y-%m-%d)}"
mkdir -p "$OUTPUT_ROOT"

QWEN_BF16="$QR/models/DeepSeek-R1-Distill-Qwen-7B"
QWEN_FP8="$QR/models/DeepSeek-R1-Distill-Qwen-7B-FP8"
LLAMA_BF16="$QR/models/DeepSeek-R1-Distill-Llama-8B"
LLAMA_FP8="$QR/models/DeepSeek-R1-Distill-Llama-8B-FP8"

echo "=== Submitting Phase 1 Matched BF16 vs FP8 Pilot Runs (n=500, seed=42) ==="

JOB_QWEN_BF16=$(
  QRM_MODEL_PATH="$QWEN_BF16" \
  QRM_OUTPUT_ROOT="$OUTPUT_ROOT" \
  QRM_MAX_SAMPLES=500 \
  QRM_SEED=42 \
  sbatch --parsable --job-name=qreason-p1-qwen7b-bf16-s42 \
    slurm/qrm_official_math500_n10.slurm
)

JOB_QWEN_FP8=$(
  QRM_MODEL_PATH="$QWEN_FP8" \
  QRM_OUTPUT_ROOT="$OUTPUT_ROOT" \
  QRM_MAX_SAMPLES=500 \
  QRM_SEED=42 \
  sbatch --parsable --job-name=qreason-p1-qwen7b-fp8-s42 \
    slurm/qrm_official_math500_n10.slurm
)

JOB_LLAMA_BF16=$(
  QRM_MODEL_PATH="$LLAMA_BF16" \
  QRM_OUTPUT_ROOT="$OUTPUT_ROOT" \
  QRM_MAX_SAMPLES=500 \
  QRM_SEED=42 \
  sbatch --parsable --job-name=qreason-p1-llama8b-bf16-s42 \
    slurm/qrm_official_math500_n10.slurm
)

JOB_LLAMA_FP8=$(
  QRM_MODEL_PATH="$LLAMA_FP8" \
  QRM_OUTPUT_ROOT="$OUTPUT_ROOT" \
  QRM_MAX_SAMPLES=500 \
  QRM_SEED=42 \
  sbatch --parsable --job-name=qreason-p1-llama8b-fp8-s42 \
    slurm/qrm_official_math500_n10.slurm
)

printf "Submitted Phase 1 Pilot Runs:\n"
printf "  Qwen-7B BF16 Job:  %s\n" "$JOB_QWEN_BF16"
printf "  Qwen-7B FP8 Job:   %s\n" "$JOB_QWEN_FP8"
printf "  Llama-8B BF16 Job: %s\n" "$JOB_LLAMA_BF16"
printf "  Llama-8B FP8 Job:  %s\n" "$JOB_LLAMA_FP8"
printf "  Output Directory:  %s\n" "$OUTPUT_ROOT"
