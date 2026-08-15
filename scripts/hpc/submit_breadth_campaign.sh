#!/usr/bin/env bash
# ==============================================================================
# Submit Breadth Evaluation Campaign (GSM8K, n=1319, Seeds 42-44)
# Evaluates Qwen-7B & Llama-8B across 4 precision formats (BF16, FP8, AWQ-4, GPTQ-4)
# Strictly adheres to QOSMaxGRESPerUser (Max 2 GPUs total: 1 Qwen + 1 Llama)
# ==============================================================================
set -euo pipefail

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"
mkdir -p logs

CAMPAIGN_DATE="${CAMPAIGN_DATE:-$(date +%Y-%m-%d)}"
OUTPUT_ROOT="${QRM_OUTPUT_ROOT:-$QR/outputs-hpc-breadth-gsm8k-${CAMPAIGN_DATE}}"
mkdir -p "$OUTPUT_ROOT"

echo "======================================================================"
echo " Launching Paper 1 Breadth Campaign Pipeline (GSM8K)"
echo " Target Output Root: $OUTPUT_ROOT"
echo "======================================================================"

# Ensure breadth datasets are ready
bash "$QR/scripts/hpc/qrm_parity/prepare_breadth_datasets.sh"

QWEN_BF16="$QR/models/DeepSeek-R1-Distill-Qwen-7B"
QWEN_FP8="$QR/models/DeepSeek-R1-Distill-Qwen-7B-FP8"
QWEN_AWQ4="$QR/models/DeepSeek-R1-Distill-Qwen-7B-AWQ-4"
QWEN_GPTQ4="$QR/models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-4"

LLAMA_BF16="$QR/models/DeepSeek-R1-Distill-Llama-8B"
LLAMA_FP8="$QR/models/DeepSeek-R1-Distill-Llama-8B-FP8"
LLAMA_AWQ4="$QR/models/DeepSeek-R1-Distill-Llama-8B-AWQ-4"
LLAMA_GPTQ4="$QR/models/DeepSeek-R1-Distill-Llama-8B-GPTQ-4"

submit_chained_job() {
  local job_name="$1"
  local model_path="$2"
  local seed="$3"
  local prev_job_id="$4"
  local max_samples="${5:-1319}"

  local dep_flag=()
  if [[ -n "$prev_job_id" && "$prev_job_id" != "none" ]]; then
    dep_flag=("--dependency=afterany:${prev_job_id}")
  fi

  local job_id
  job_id=$(
    QRM_MODEL_PATH="$model_path" \
    QRM_OUTPUT_ROOT="$OUTPUT_ROOT" \
    QRM_MAX_SAMPLES="$max_samples" \
    QRM_SEED="$seed" \
    sbatch --parsable \
      --job-name="$job_name" \
      "${dep_flag[@]}" \
      slurm/qrm_official_math500_n10.slurm
  )
  echo "$job_id"
}

echo ""
echo "Breadth campaign launcher configured. Ready to run when authorized."
echo "Use: bash scripts/hpc/submit_breadth_campaign.sh"
