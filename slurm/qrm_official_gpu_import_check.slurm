#!/bin/bash
#SBATCH --job-name=qrm-gpu-import-check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/%u/reasoning-compression-lab/logs/qrm_gpu_check_%j.out
#SBATCH --error=/scratch/%u/reasoning-compression-lab/logs/qrm_gpu_check_%j.err

set -eo pipefail

QR="/scratch/${USER}/reasoning-compression-lab"
QRM_DIR="$QR/external/Quantized-Reasoning-Models"
cd "$QR"

echo "=== QRM GPU import check (vllm._C + CUDA platform) ==="
bash scripts/hpc/qrm_parity/verify_qrm_official_preflight.sh

source /home/apps/MSCC/miniconda3/etc/profile.d/conda.sh
conda activate qrm-official
export PATH="$CONDA_PREFIX/bin:/usr/bin:/bin:$PATH"
export GIT_PYTHON_GIT_EXECUTABLE="$CONDA_PREFIX/bin/git"
cd "$QRM_DIR"

nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1

python - <<'PY'
import importlib
import vllm
print(f"vllm {vllm.__version__}")
importlib.import_module("vllm.platforms.cuda")
print("vllm.platforms.cuda: ok")
import vllm._C  # noqa: F401
print("vllm._C: ok")
from lighteval.logging.evaluation_tracker import EvaluationTracker  # noqa: F401
print("lighteval EvaluationTracker: ok")
print("GPU_IMPORT_CHECK_OK")
PY

echo "=== GPU import check passed ==="