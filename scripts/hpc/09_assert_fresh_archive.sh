#!/usr/bin/env bash
# Fail fast if publication output root is unsafe (bad archive or stale partials).
#
# Usage (on HPC, before b01):
#   export QR=/scratch/$USER/reasoning-compression-lab
#   export QREASON_OUTPUT_ROOT=$QR/outputs-hpc-2a100-main-$(date +%Y-%m-%d)-rerun
#   bash scripts/hpc/09_assert_fresh_archive.sh
set -euo pipefail

QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
cd "$QR"

ROOT="${QREASON_OUTPUT_ROOT:-}"
if [[ -z "$ROOT" ]]; then
  echo "ERROR: set QREASON_OUTPUT_ROOT before running publication blocks." >&2
  exit 1
fi

if [[ "$ROOT" == *"outputs-hpc-2a100-main-2026-06-29"* ]]; then
  if [[ "${QREASON_ALLOW_BAD_ARCHIVE:-}" != "1" ]]; then
    echo "ERROR: QREASON_OUTPUT_ROOT points at the invalid June-29 archive." >&2
    echo "  rm -rf outputs-hpc-2a100-main-2026-06-29" >&2
    echo "  export QREASON_OUTPUT_ROOT=\$QR/outputs-hpc-2a100-main-\$(date +%Y-%m-%d)-rerun" >&2
    exit 1
  fi
  echo "WARN: QREASON_ALLOW_BAD_ARCHIVE=1 — proceeding at your own risk." >&2
fi

if [[ "${QREASON_FRESH_RUN:-}" == "1" && -d "$ROOT/raw" ]]; then
  count="$(find "$ROOT/raw" -name '*.jsonl' 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$count" -gt 0 && "${QREASON_ALLOW_RESUME:-}" != "1" ]]; then
    echo "ERROR: QREASON_FRESH_RUN=1 but $ROOT/raw already has $count JSONL file(s)." >&2
    echo "  rm -rf \"$ROOT\" or unset QREASON_FRESH_RUN" >&2
    exit 1
  fi
fi

python3 scripts/hpc/09_assert_fresh_archive.py --archive "$ROOT"
echo "Archive check passed: $ROOT"
