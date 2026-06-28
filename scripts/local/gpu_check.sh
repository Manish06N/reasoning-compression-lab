#!/usr/bin/env bash
# Quick GPU + vLLM sanity check on WSL (RTX 5080).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$QR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qreason

echo "=== nvidia-smi ==="
nvidia-smi

echo ""
echo "=== Python / torch / vLLM ==="
python - <<'PY'
import torch
import vllm
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("vllm:", vllm.__version__)
PY

echo ""
echo "GPU check passed."
