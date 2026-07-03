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

param_rudra_configure_triton_cc() {
  # Compute nodes often lack /usr/include (e.g. stdlib.h). Triton JIT uses $CC or `which gcc`;
  # param_rudra_activate_conda prepends /usr/bin to PATH, so system gcc wins without CC set.
  local conda_gcc="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
  local conda_gxx="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
  if [[ -x "$conda_gcc" ]]; then
    export CC="$conda_gcc"
    export CXX="$conda_gxx"
    return 0
  fi
  echo "WARN: conda gcc not found at $conda_gcc; Triton JIT may fail on compute nodes." >&2
  echo "  Fix: conda activate qreason && conda install -y -c conda-forge gcc_linux-64 gxx_linux-64 sysroot_linux-64" >&2
  return 1
}

param_rudra_assert_triton_cc() {
  param_rudra_configure_triton_cc || return 1
  local probe
  probe="$(mktemp -t triton_cc_probe.XXXXXX.c)"
  echo '#include <stdlib.h>' >"$probe"
  if ! "$CC" -c "$probe" -o "${probe%.c}.o" 2>/dev/null; then
    rm -f "$probe" "${probe%.c}.o"
    echo "ERROR: Triton host compiler check failed (CC=$CC cannot compile stdlib.h)." >&2
    echo "  Re-run: bash scripts/hpc/00_setup_env.sh" >&2
    return 1
  fi
  rm -f "$probe" "${probe%.c}.o"
}

param_rudra_activate_conda() {
  if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: conda not found at $CONDA_ROOT"
    return 1
  fi
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate qreason
  export PATH="/usr/bin:/bin:${PATH}"
  if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git not found after conda activate qreason." >&2
    echo "  Fix: conda activate qreason && conda install -y git" >&2
    echo "  Or re-run: bash scripts/hpc/00_setup_env.sh" >&2
    return 1
  fi
  param_rudra_configure_triton_cc || true
}
