#!/usr/bin/env bash
# Submit measured serving benchmark jobs to SLURM on PARAM Rudra HPC.
# Strict rule: 1 GPU per cell, max 2 concurrent jobs, qrm-official conda env.

set -euo pipefail

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
BENCHMARK_SCRIPT="$QR/scripts/hpc/qrm_parity/benchmark_serving.py"
LOGS_DIR="$QR/logs/measured_serving"
mkdir -p "$LOGS_DIR"

PYTHON_BIN="/home/manishn_iitp/.conda/envs/qrm-official/bin/python3"
MODELS_DIR="$QR/models"

# Define 8 configurations
CONFIGS=(
  "Qwen-7B:BF16:$MODELS_DIR/DeepSeek-R1-Distill-Qwen-7B"
  "Qwen-7B:FP8:$MODELS_DIR/DeepSeek-R1-Distill-Qwen-7B-FP8"
  "Qwen-7B:AWQ-4:$MODELS_DIR/DeepSeek-R1-Distill-Qwen-7B-AWQ-4"
  "Qwen-7B:GPTQ-4:$MODELS_DIR/DeepSeek-R1-Distill-Qwen-7B-GPTQ-4"
  "Llama-8B:BF16:$MODELS_DIR/DeepSeek-R1-Distill-Llama-8B"
  "Llama-8B:FP8:$MODELS_DIR/DeepSeek-R1-Distill-Llama-8B-FP8"
  "Llama-8B:AWQ-4:$MODELS_DIR/DeepSeek-R1-Distill-Llama-8B-AWQ-4"
  "Llama-8B:GPTQ-4:$MODELS_DIR/DeepSeek-R1-Distill-Llama-8B-GPTQ-4"
)

# Chained submissions: 2 parallel pipelines (Channel A: Qwen-7B, Channel B: Llama-8B)
prev_qwen_jobid=""
prev_llama_jobid=""

for cfg in "${CONFIGS[@]}"; do
  IFS=':' read -r m_name fmt m_path <<< "$cfg"
  job_name="srv-${m_name,,}-${fmt,,}"
  log_file="$LOGS_DIR/${m_name}_${fmt}.out"
  err_file="$LOGS_DIR/${m_name}_${fmt}.err"

  dep_arg=""
  if [[ "$m_name" == "Qwen-7B" && -n "$prev_qwen_jobid" ]]; then
    dep_arg="--dependency=afterany:$prev_qwen_jobid"
  elif [[ "$m_name" == "Llama-8B" && -n "$prev_llama_jobid" ]]; then
    dep_arg="--dependency=afterany:$prev_llama_jobid"
  fi

  sbatch_cmd=(
    sbatch
    --job-name="$job_name"
    --partition=gpu
    --gres=gpu:1
    --cpus-per-task=16
    --time=47:00:00
    --output="$log_file"
    --error="$err_file"
  )
  if [[ -n "$dep_arg" ]]; then
    sbatch_cmd+=("$dep_arg")
  fi

  job_id=$(
    "${sbatch_cmd[@]}" --wrap="
      export PYTHONPATH='$QR:$QR/external/Quantized-Reasoning-Models:$QR/external/Quantized-Reasoning-Models/third-party/lighteval/src:\$PYTHONPATH'
      export PYTHONUNBUFFERED=1
      export VLLM_WORKER_MULTIPROC_METHOD=spawn
      export TORCH_COMPILE_DISABLE=1
      export TORCHDYNAMO_DISABLE=1
      $PYTHON_BIN -u '$BENCHMARK_SCRIPT' \
        --model-path '$m_path' \
        --model-name '$m_name' \
        --format '$fmt' \
        --repetitions 3
    " | awk '{print $NF}'
  )

  echo "Submitted $m_name $fmt -> Job ID: $job_id (Dependency: ${dep_arg:-none})"

  if [[ "$m_name" == "Qwen-7B" ]]; then
    prev_qwen_jobid="$job_id"
  else
    prev_llama_jobid="$job_id"
  fi
done

echo "All 8 serving benchmark jobs submitted across 2 parallel pipelines."
