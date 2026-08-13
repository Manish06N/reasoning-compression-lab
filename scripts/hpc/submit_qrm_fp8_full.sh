#!/usr/bin/env bash
# Submit full FP8 MATH-500 runs only after both exact-stack n=10 gates pass.
set -euo pipefail

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"

VALIDATION_ROOT="${QRM_VALIDATION_ROOT:-$QR/outputs-hpc-qrm-official-fp8-validation-2026-08-13}"
QWEN_MODEL="$QR/models/DeepSeek-R1-Distill-Qwen-7B-FP8"
LLAMA_MODEL="$QR/models/DeepSeek-R1-Distill-Llama-8B-FP8"
QWEN_RESULT="$VALIDATION_ROOT/inference/DeepSeek-R1-Distill-Qwen-7B-FP8-seed42/MATH-500.jsonl"
LLAMA_RESULT="$VALIDATION_ROOT/inference/DeepSeek-R1-Distill-Llama-8B-FP8-seed42/MATH-500.jsonl"
QRM_PYTHON="${QRM_PYTHON:-/home/manishn_iitp/.conda/envs/qrm-official/bin/python3}"

validate_pilot() {
  local model="$1"
  local result="$2"
  "$QRM_PYTHON" scripts/hpc/qrm_parity/validate_official_results.py \
    --result "$result" \
    --model "$model" \
    --expected-rows 10 \
    --min-accuracy 1.0 \
    --min-boxed-rate 1.0 \
    --max-new-tokens 32768 \
    --max-token-limit-hits 0 \
    --max-repetition-rows 0
}

echo "=== Strict pre-submit gate: Qwen-7B FP8 ==="
validate_pilot "$QWEN_MODEL" "$QWEN_RESULT"
echo "=== Strict pre-submit gate: Llama-8B FP8 ==="
validate_pilot "$LLAMA_MODEL" "$LLAMA_RESULT"

OUTPUT_ROOT="${QRM_OUTPUT_ROOT:-$QR/outputs-hpc-qrm-official-fp8-full-$(date +%Y-%m-%d-%H%M%S)}"
mkdir -p "$OUTPUT_ROOT" logs

QWEN_JOB_ID=$(
  QRM_MODEL_PATH="$QWEN_MODEL" \
  QRM_OUTPUT_ROOT="$OUTPUT_ROOT" \
  QRM_MAX_SAMPLES=500 \
  QRM_SEED=42 \
  sbatch --parsable --job-name=qreason-qrm-fp8-qwen-n500 \
    slurm/qrm_official_math500_n10.slurm
)
LLAMA_JOB_ID=$(
  QRM_MODEL_PATH="$LLAMA_MODEL" \
  QRM_OUTPUT_ROOT="$OUTPUT_ROOT" \
  QRM_MAX_SAMPLES=500 \
  QRM_SEED=42 \
  sbatch --parsable --job-name=qreason-qrm-fp8-llama-n500 \
    slurm/qrm_official_math500_n10.slurm
)

printf 'Submitted exact-stack FP8 full runs:\n'
printf '  Qwen:  %s\n' "$QWEN_JOB_ID"
printf '  Llama: %s\n' "$LLAMA_JOB_ID"
printf '  Output: %s\n' "$OUTPUT_ROOT"
