#!/usr/bin/env bash
set -euo pipefail

export QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qreason

echo "=== nvidia-smi ==="
nvidia-smi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

python - <<'PY'
try:
    import vllm
    print("vllm:", vllm.__version__)
except Exception as exc:
    print("vllm import failed:", exc)
    raise
PY

echo "GPU check passed."
