#!/usr/bin/env bash
# PARAM Rudra (IIT Patna) environment — sourced by scripts/hpc/*.sh on the cluster.

export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"

# Keep Hugging Face cache on scratch (not home).
export HF_HOME="${HF_HOME:-${QR}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Model path default for Level A.
export QREASON_MODEL_QWEN7B="${QREASON_MODEL_QWEN7B:-${QR}/models/DeepSeek-R1-Distill-Qwen-7B}"

# PARAM Rudra: conda lives here (see CODEX.md), not via module load.
CONDA_ROOT="${CONDA_ROOT:-/home/apps/MSCC/miniconda3}"

export QREASON_MODEL_LLAMA8B="${QREASON_MODEL_LLAMA8B:-${QR}/models/DeepSeek-R1-Distill-Llama-8B}"
export QREASON_MODEL_LLAMA8B_FP8="${QREASON_MODEL_LLAMA8B_FP8:-${QR}/models/DeepSeek-R1-Distill-Llama-8B-FP8}"
export QREASON_MODEL_LLAMA8B_AWQ4="${QREASON_MODEL_LLAMA8B_AWQ4:-${QR}/models/DeepSeek-R1-Distill-Llama-8B-AWQ-4}"
export QREASON_MODEL_LLAMA8B_GPTQ4="${QREASON_MODEL_LLAMA8B_GPTQ4:-${QR}/models/DeepSeek-R1-Distill-Llama-8B-GPTQ-4}"

param_rudra_activate_conda() {
  if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: conda not found at $CONDA_ROOT"
    return 1
  fi
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate qreason
}
