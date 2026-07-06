#!/usr/bin/env bash
# Install official QRM evaluation stack in a dedicated conda env (separate from qreason).
set -eo pipefail

# Compute nodes may hide system git/gcc after conda activate (see KNOWN_ISSUES §3b).
export PATH="/usr/bin:/bin:${PATH}"

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
QRM_DIR="${QRM_REPO_DIR:-$QR/external/Quantized-Reasoning-Models}"
ENV_NAME="${QRM_CONDA_ENV:-qrm-official}"
CONDA_ROOT="${CONDA_ROOT:-/home/apps/MSCC/miniconda3}"
MARKER="$QR/.qrm_official_env_ready"
INSTALL_REV="2026-07-06-conda-gcc12-nvcc124-vllm070wheel"
VLLM_PRECOMPILED_WHEEL_URL="https://files.pythonhosted.org/packages/51/70/6fc00dca2e9f53a76b7792d788cb2efbb9d2587ed0ca9a71d5ccf7fc7543/vllm-0.7.0-cp38-abi3-manylinux1_x86_64.whl"

qrm_export_cuda_build_env() {
  local pyver site_pkgs inc_paths=("$CONDA_PREFIX/include") d joined
  pyver="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  site_pkgs="$CONDA_PREFIX/lib/python${pyver}/site-packages/nvidia"
  if [[ -d "$site_pkgs" ]]; then
    for d in "$site_pkgs"/*/include; do
      [[ -d "$d" ]] && inc_paths+=("$d")
    done
  fi
  joined="$(IFS=:; echo "${inc_paths[*]}")"
  export CPATH="$joined${CPATH:+:$CPATH}"
  export CPLUS_INCLUDE_PATH="$joined${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
}

if [[ ! -d "$QRM_DIR/.git" ]]; then
  bash "$QR/scripts/hpc/qrm_parity/setup_official_qrm_repo.sh"
fi

cd "$QRM_DIR"
HAD_DIR="$QRM_DIR/third-party/fast-hadamard-transform"
if [[ ! -f "$HAD_DIR/setup.py" && ! -f "$HAD_DIR/pyproject.toml" ]]; then
  if command -v git >/dev/null 2>&1; then
    echo "=== Initializing QRM submodules (lighteval, vllm, fast-hadamard-transform) ==="
    git submodule update --init --recursive \
      third-party/lighteval \
      third-party/vllm \
      third-party/fast-hadamard-transform
  else
    echo "ERROR: fast-hadamard-transform submodule missing and git unavailable on this node." >&2
    echo "  Run on login node: bash scripts/hpc/qrm_parity/setup_official_qrm_repo.sh" >&2
    exit 1
  fi
else
  echo "=== QRM submodules present (init skipped) ==="
fi

if [[ -f "$MARKER" ]] && grep -qxF "$INSTALL_REV" "$MARKER"; then
  echo "QRM official env already installed ($MARKER: $INSTALL_REV). Skipping pip install."
  exit 0
fi

if [[ -f "$MARKER" ]]; then
  echo "WARN: stale QRM env marker ($(cat "$MARKER")); reinstalling stack." >&2
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env $ENV_NAME exists"
else
  conda create -n "$ENV_NAME" python=3.11 -y
fi
conda activate "$ENV_NAME"
export PATH="/usr/bin:/bin:${PATH}"

pip install -U pip wheel setuptools

echo "=== Installing QRM base requirements (pins torch 2.5.1; overwrites bad pip vllm fallback) ==="
pip install -r "$QRM_DIR/requirements.txt"
# Remove packages left by the failed pip vllm==0.8.5 fallback (job 87130).
pip uninstall -y vllm torchvision torchaudio xformers 2>/dev/null || true

echo "=== Ensuring host + CUDA toolchain (compute nodes lack g++; nvcc 12.4 needs gcc<=13) ==="
conda install -y -c conda-forge -c nvidia \
  gcc_linux-64=12 gxx_linux-64=12 sysroot_linux-64 git \
  cuda-nvcc=12.4 cuda-cudart-dev=12.4 cuda-cudart=12.4 cuda-cccl=12.4

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export PATH="$CONDA_PREFIX/bin:$PATH"
qrm_export_cuda_build_env

if [[ ! -x "$CC" || ! -x "$CXX" ]]; then
  echo "ERROR: conda host compilers missing (CC=$CC CXX=$CXX)" >&2
  exit 1
fi
probe="$(mktemp -t qrm_cc_probe.XXXXXX.c)"
echo '#include <stdlib.h>' >"$probe"
if ! "$CC" -c "$probe" -o "${probe%.c}.o"; then
  rm -f "$probe" "${probe%.c}.o"
  echo "ERROR: host compiler check failed (CC=$CC cannot compile stdlib.h)" >&2
  exit 1
fi
rm -f "$probe" "${probe%.c}.o"
echo "Using host CC: $CC"
echo "Using host CXX: $CXX"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc still missing after conda cuda-nvcc install (CUDA_HOME=$CUDA_HOME)" >&2
  exit 1
fi
echo "Using nvcc: $(command -v nvcc)"
echo "CUDA include paths: ${CPATH%%:*}:..."

echo "=== Installing fast-hadamard-transform (QRM submodule) ==="
pip install --no-build-isolation -e "$QRM_DIR/third-party/fast-hadamard-transform"

echo "=== Installing lighteval (QRM fork) ==="
pip install -e "$QRM_DIR/third-party/lighteval[math]"

echo "=== Installing QRM vLLM fork v0.7.0 (official PyPI precompiled wheel) ==="
rm -f "$QRM_DIR/third-party/vllm/vllm/_C.abi3.so"
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION="$VLLM_PRECOMPILED_WHEEL_URL"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.7.0
pip install -e "$QRM_DIR/third-party/vllm" || {
  echo "WARN: editable vllm failed; retrying with --no-build-isolation" >&2
  pip install --no-build-isolation -e "$QRM_DIR/third-party/vllm"
}

pip install typer rich

echo "=== Verifying QRM stack imports ==="
python - <<'PY'
import fast_hadamard_transform  # noqa: F401
from vllm_custom.model_executor.fake_quantized_models.registry import register_fake_quantized_models
register_fake_quantized_models()
import vllm
print("fast_hadamard_transform: ok")
print("vllm_custom registry: ok")
print(f"vllm version: {getattr(vllm, '__version__', 'unknown')}")
try:
    import vllm._C  # noqa: F401
    print("vllm._C: ok")
except ImportError as exc:
    msg = str(exc)
    if "libcuda.so.1" in msg or "libcuda.so" in msg:
        print("vllm._C: deferred on login node (no GPU driver); GPU job will verify")
    elif "undefined symbol" in msg:
        raise RuntimeError(f"vllm._C ABI mismatch: {exc}") from exc
    else:
        raise
PY

printf '%s\n' "$INSTALL_REV" >"$MARKER"
echo "QRM official env ready: conda activate $ENV_NAME"