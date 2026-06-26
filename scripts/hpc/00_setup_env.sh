#!/usr/bin/env bash
# Run on HPC login node (or inside an interactive GPU session after modules load).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=param_rudra_env.sh
source "${SCRIPT_DIR}/param_rudra_env.sh"

cd "$QR"
echo "Workspace: $QR"
echo "HF cache: $HF_HOME"

param_rudra_load_modules

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Run: module load mldl/Miniconda"
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx qreason; then
  echo "Conda env qreason already exists."
else
  conda create -n qreason python=3.11 -y
fi

param_rudra_activate_conda

pip install --upgrade pip
pip install -r requirements-hpc.txt

echo "Environment ready. Test with: bash scripts/hpc/01_gpu_check.sh"
