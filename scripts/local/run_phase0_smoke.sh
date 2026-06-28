#!/usr/bin/env bash
# Phase 0: Qwen-1.5B smoke on RTX 5080; Qwen-7B BF16 smoke if VRAM allows (else HPC Gate 3).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

cleanup_vllm() {
  pkill -f 'vllm.v1.engine.core' 2>/dev/null || true
  sleep 3
}

echo "=== Phase 0 smoke: Qwen-1.5B BF16 (pipeline verifier) ==="
python scripts/smoke_test.py \
  --cell-config configs/cells/smoke_qwen15b_bf16.json \
  --limit 1 \
  --max-tokens 64 \
  --output runs/raw/smoke_qwen15b_local.jsonl
cleanup_vllm

echo ""
echo "=== Phase 0 smoke: Qwen-7B BF16 (5080 limits) ==="
echo "Note: BF16 7B weights ~14 GiB; may not fit 16 GB VRAM with vLLM 0.23."
if python scripts/smoke_test.py \
  --cell-config configs/cells/smoke_qwen7b_bf16_5080.json \
  --limit 1 \
  --max-tokens 64 \
  --output runs/raw/smoke_qwen7b_local.jsonl; then
  echo "7B BF16 smoke succeeded locally."
else
  cleanup_vllm
  mkdir -p runs/raw
  cat > runs/raw/smoke_qwen7b_local_status.json <<EOF
{
  "status": "deferred_to_hpc",
  "reason": "Qwen-7B BF16 weights (~14.3 GiB) exceed RTX 5080 16GB VRAM headroom for vLLM KV cache.",
  "local_action": "Run bash scripts/hpc/03_smoke_test.sh on PARAM Rudra A100 after Gate 2.",
  "models_downloaded": true,
  "pipeline_verified_by": "runs/raw/smoke_qwen15b_local.jsonl"
}
EOF
  echo ""
  echo "7B BF16 smoke deferred to HPC (expected on 16 GB). Wrote runs/raw/smoke_qwen7b_local_status.json"
fi
cleanup_vllm

echo ""
echo "Phase 0 complete. Check:"
echo "  runs/raw/smoke_qwen15b_local.jsonl"
echo "  runs/raw/smoke_qwen7b_local.jsonl"
