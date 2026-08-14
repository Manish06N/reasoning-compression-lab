#!/usr/bin/env bash
# ==============================================================================
# Submit Complete Publication Campaign Pipeline (MATH-500, Seeds 42-44)
# Implements two parallel chained pipelines (Qwen Channel & Llama Channel)
# Strictly adheres to QOSMaxGRESPerUser (Max 2 GPUs total: 1 Qwen + 1 Llama)
# ==============================================================================
set -euo pipefail

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"
mkdir -p logs

CAMPAIGN_DATE="${CAMPAIGN_DATE:-$(date +%Y-%m-%d)}"
OUTPUT_ROOT="${QRM_OUTPUT_ROOT:-$QR/outputs-hpc-campaign-${CAMPAIGN_DATE}}"
mkdir -p "$OUTPUT_ROOT"

echo "======================================================================"
echo " Launching Paper 1 Publication Campaign Pipeline"
echo " Target Output Root: $OUTPUT_ROOT"
echo "======================================================================"

# Model paths
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
  local max_samples="${5:-500}"

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
echo "--- Submitting Channel 1: Qwen-7B Chained Pipeline (1 GPU) ---"

# Phase 1: Seed 42 Matched Anchors
Q1=$(submit_chained_job "p1-qwen7b-bf16-s42" "$QWEN_BF16" 42 "none")
echo "  [Qwen 1] BF16  (Seed 42): $Q1 (Initial Active)"

Q2=$(submit_chained_job "p1-qwen7b-fp8-s42" "$QWEN_FP8" 42 "$Q1")
echo "  [Qwen 2] FP8   (Seed 42): $Q2 (Chained after $Q1)"

Q3=$(submit_chained_job "p2-qwen7b-awq4-s42" "$QWEN_AWQ4" 42 "$Q2")
echo "  [Qwen 3] AWQ-4 (Seed 42): $Q3 (Chained after $Q2)"

Q4=$(submit_chained_job "p2-qwen7b-gptq4-s42" "$QWEN_GPTQ4" 42 "$Q3")
echo "  [Qwen 4] GPTQ4 (Seed 42): $Q4 (Chained after $Q3)"

# Phase 2: Seed 43 Pilot
Q5=$(submit_chained_job "p2-qwen7b-bf16-s43" "$QWEN_BF16" 43 "$Q4")
echo "  [Qwen 5] BF16  (Seed 43): $Q5 (Chained after $Q4)"

Q6=$(submit_chained_job "p2-qwen7b-fp8-s43" "$QWEN_FP8" 43 "$Q5")
echo "  [Qwen 6] FP8   (Seed 43): $Q6 (Chained after $Q5)"

Q7=$(submit_chained_job "p2-qwen7b-awq4-s43" "$QWEN_AWQ4" 43 "$Q6")
echo "  [Qwen 7] AWQ-4 (Seed 43): $Q7 (Chained after $Q6)"

Q8=$(submit_chained_job "p2-qwen7b-gptq4-s43" "$QWEN_GPTQ4" 43 "$Q7")
echo "  [Qwen 8] GPTQ4 (Seed 43): $Q8 (Chained after $Q7)"

# Phase 2: Seed 44 Pilot
Q9=$(submit_chained_job "p2-qwen7b-bf16-s44" "$QWEN_BF16" 44 "$Q8")
echo "  [Qwen 9] BF16  (Seed 44): $Q9 (Chained after $Q8)"

Q10=$(submit_chained_job "p2-qwen7b-fp8-s44" "$QWEN_FP8" 44 "$Q9")
echo "  [Qwen 10] FP8  (Seed 44): $Q10 (Chained after $Q9)"

Q11=$(submit_chained_job "p2-qwen7b-awq4-s44" "$QWEN_AWQ4" 44 "$Q10")
echo "  [Qwen 11] AWQ4 (Seed 44): $Q11 (Chained after $Q10)"

Q12=$(submit_chained_job "p2-qwen7b-gptq4-s44" "$QWEN_GPTQ4" 44 "$Q11")
echo "  [Qwen 12] GPTQ4 (Seed 44): $Q12 (Chained after $Q11)"


echo ""
echo "--- Submitting Channel 2: Llama-8B Chained Pipeline (1 GPU) ---"

# Phase 1: Seed 42 Matched Anchors
L1=$(submit_chained_job "p1-llama8b-bf16-s42" "$LLAMA_BF16" 42 "none")
echo "  [Llama 1] BF16  (Seed 42): $L1 (Initial Active)"

L2=$(submit_chained_job "p1-llama8b-fp8-s42" "$LLAMA_FP8" 42 "$L1")
echo "  [Llama 2] FP8   (Seed 42): $L2 (Chained after $L1)"

L3=$(submit_chained_job "p2-llama8b-awq4-s42" "$LLAMA_AWQ4" 42 "$L2")
echo "  [Llama 3] AWQ-4 (Seed 42): $L3 (Chained after $L2)"

L4=$(submit_chained_job "p2-llama8b-gptq4-s42" "$LLAMA_GPTQ4" 42 "$L3")
echo "  [Llama 4] GPTQ4 (Seed 42): $L4 (Chained after $L3)"

# Phase 2: Seed 43 Pilot
L5=$(submit_chained_job "p2-llama8b-bf16-s43" "$LLAMA_BF16" 43 "$L4")
echo "  [Llama 5] BF16  (Seed 43): $L5 (Chained after $L4)"

L6=$(submit_chained_job "p2-llama8b-fp8-s43" "$LLAMA_FP8" 43 "$L5")
echo "  [Llama 6] FP8   (Seed 43): $L6 (Chained after $L5)"

L7=$(submit_chained_job "p2-llama8b-awq4-s43" "$LLAMA_AWQ4" 43 "$L6")
echo "  [Llama 7] AWQ-4 (Seed 43): $L7 (Chained after $L6)"

L8=$(submit_chained_job "p2-llama8b-gptq4-s43" "$LLAMA_GPTQ4" 43 "$L7")
echo "  [Llama 8] GPTQ4 (Seed 43): $L8 (Chained after $L7)"

# Phase 2: Seed 44 Pilot
L9=$(submit_chained_job "p2-llama8b-bf16-s44" "$LLAMA_BF16" 44 "$L8")
echo "  [Llama 9] BF16  (Seed 44): $L9 (Chained after $L8)"

L10=$(submit_chained_job "p2-llama8b-fp8-s44" "$LLAMA_FP8" 44 "$L9")
echo "  [Llama 10] FP8 (Seed 44): $L10 (Chained after $L9)"

L11=$(submit_chained_job "p2-llama8b-awq4-s44" "$LLAMA_AWQ4" 44 "$L10")
echo "  [Llama 11] AWQ4 (Seed 44): $L11 (Chained after $L10)"

L12=$(submit_chained_job "p2-llama8b-gptq4-s44" "$LLAMA_GPTQ4" 44 "$L11")
echo "  [Llama 12] GPTQ4 (Seed 44): $L12 (Chained after $L11)"

ALL_JOB_IDS="$Q1 $Q2 $Q3 $Q4 $Q5 $Q6 $Q7 $Q8 $Q9 $Q10 $Q11 $Q12 $L1 $L2 $L3 $L4 $L5 $L6 $L7 $L8 $L9 $L10 $L11 $L12"
echo "$ALL_JOB_IDS" > "$OUTPUT_ROOT/pipeline_jobs.txt"

echo ""
echo "======================================================================"
echo " All 24 publication cells submitted and chained!"
echo " Max concurrent active GPUs: 2 (1 Qwen + 1 Llama)"
echo " Saved job manifest to: $OUTPUT_ROOT/pipeline_jobs.txt"
echo "======================================================================"
