#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=param_rudra_env.sh
source "${SCRIPT_DIR}/param_rudra_env.sh"

cd "$QR"
param_rudra_activate_conda

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
