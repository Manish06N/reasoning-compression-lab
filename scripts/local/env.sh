#!/usr/bin/env bash
# Source from WSL before running local experiments:
#   source scripts/local/env.sh
export QR="/mnt/g/ALL MY Projects/2026/03-paper1-experiments"
cd "$QR"

export HF_HOME="${HF_HOME:-$QR/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$QR/models" "$QR/runs/raw" "$QR/results"

if [[ -f "$QR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source <(tr -d '\r' < "$QR/.env")
  set +a
fi

# Strip Windows CRLF artifacts when .env was edited on Windows.
for _var in QR HF_TOKEN QREASON_MODEL_QWEN15B QREASON_MODEL_QWEN15B_FP8 \
  QREASON_MODEL_QWEN15B_AWQ4 QREASON_MODEL_QWEN15B_GPTQ4 \
  QREASON_MODEL_QWEN7B QREASON_MODEL_QWEN7B_FP8 QREASON_MODEL_QWEN7B_AWQ4 \
  QREASON_MODEL_QWEN7B_GPTQ4 QREASON_MODEL_QWEN7B_GPTQ3 \
  QREASON_MODEL_LLAMA8B QREASON_MODEL_LLAMA8B_FP8 \
  QREASON_MODEL_LLAMA8B_AWQ4 QREASON_MODEL_LLAMA8B_GPTQ4; do
  if [[ -n "${!_var:-}" ]]; then
    # shellcheck disable=SC2154
    printf -v "$_var" '%s' "$(printf '%s' "${!_var}" | tr -d '\r')"
    export "$_var"
  fi
done
cd "$QR"

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  printf '%s' "$HF_TOKEN" > "$HF_HOME/token"
  chmod 600 "$HF_HOME/token"
elif [[ -f "$HF_HOME/token" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$HF_HOME/token")"
  export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qreason

# vLLM 0.23+ wheels link against CUDA 13 libs shipped via pip (nvidia-cu13).
_cuda13_lib="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13/lib"
_cuda12_lib="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
if [[ -d "$_cuda13_lib" ]]; then
  export LD_LIBRARY_PATH="$_cuda13_lib:$_cuda12_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset _cuda13_lib _cuda12_lib

# Blackwell (sm_120) / WSL: FlashInfer JIT fails arch check until CUDA 12.9+ runtime.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

export QREASON_MODEL_QWEN15B="${QREASON_MODEL_QWEN15B:-$QR/models/DeepSeek-R1-Distill-Qwen-1.5B}"
export QREASON_MODEL_QWEN15B_FP8="${QREASON_MODEL_QWEN15B_FP8:-$QR/models/DeepSeek-R1-Distill-Qwen-1.5B-FP8}"
export QREASON_MODEL_QWEN15B_AWQ4="${QREASON_MODEL_QWEN15B_AWQ4:-$QR/models/DeepSeek-R1-Distill-Qwen-1.5B-AWQ-4}"
export QREASON_MODEL_QWEN15B_GPTQ4="${QREASON_MODEL_QWEN15B_GPTQ4:-$QR/models/DeepSeek-R1-Distill-Qwen-1.5B-GPTQ-4}"
export QREASON_MODEL_QWEN7B="${QREASON_MODEL_QWEN7B:-$QR/models/DeepSeek-R1-Distill-Qwen-7B}"
export QREASON_MODEL_QWEN7B_FP8="${QREASON_MODEL_QWEN7B_FP8:-$QR/models/DeepSeek-R1-Distill-Qwen-7B-FP8}"
export QREASON_MODEL_QWEN7B_AWQ4="${QREASON_MODEL_QWEN7B_AWQ4:-$QR/models/DeepSeek-R1-Distill-Qwen-7B-AWQ-4}"
export QREASON_MODEL_QWEN7B_GPTQ4="${QREASON_MODEL_QWEN7B_GPTQ4:-$QR/models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-4}"
export QREASON_MODEL_QWEN7B_GPTQ3="${QREASON_MODEL_QWEN7B_GPTQ3:-$QR/models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-3}"
export QREASON_MODEL_LLAMA8B="${QREASON_MODEL_LLAMA8B:-$QR/models/DeepSeek-R1-Distill-Llama-8B}"
export QREASON_MODEL_LLAMA8B_FP8="${QREASON_MODEL_LLAMA8B_FP8:-$QR/models/DeepSeek-R1-Distill-Llama-8B-FP8}"
export QREASON_MODEL_LLAMA8B_AWQ4="${QREASON_MODEL_LLAMA8B_AWQ4:-$QR/models/DeepSeek-R1-Distill-Llama-8B-AWQ-4}"
export QREASON_MODEL_LLAMA8B_GPTQ4="${QREASON_MODEL_LLAMA8B_GPTQ4:-$QR/models/DeepSeek-R1-Distill-Llama-8B-GPTQ-4}"

echo "QR=$QR"
echo "conda env: qreason"
echo "model path (7B): $QREASON_MODEL_QWEN7B"
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF auth: configured"
else
  echo "HF auth: missing (set HF_TOKEN in .env or copy hf_cache/token from HPC)"
fi
