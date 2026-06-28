#!/usr/bin/env bash
# RTX 5080 (Blackwell sm_120) requires PyTorch built with CUDA 12.8+.
# Run once in WSL: bash scripts/local/upgrade_pytorch_blackwell.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

echo "Current torch:"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

pip install --upgrade pip
pip install --upgrade --force-reinstall torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
pip install --upgrade "vllm>=0.23.0" "numpy<2.4"

echo ""
echo "Upgraded torch:"
python -c "import torch; print(torch.__version__); print('capability:', torch.cuda.get_device_capability(0)); x=torch.zeros(1,device='cuda'); print('cuda tensor ok')"
