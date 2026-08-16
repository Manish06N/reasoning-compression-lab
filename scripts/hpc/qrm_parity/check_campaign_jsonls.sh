#!/usr/bin/env bash
# Check whether the 56k campaign JSONLs still exist on PARAM Rudra, then export
# answer-like fields to a *separate* directory. CPU only. Read-only on the
# campaign tree. Run on the cluster, not on the MacBook.
#
#   export QR=/scratch/manishn_iitp/reasoning-compression-lab
#   bash $QR/scripts/hpc/qrm_parity/check_campaign_jsonls.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QR="${QR:-/scratch/${USER}/reasoning-compression-lab}"
JSONL_ROOT="${1:-$QR/outputs-hpc-campaign-2026-08-14/inference}"
OUT="${2:-$QR/results/extracted_answers}"
EXPORTER="$SCRIPT_DIR/export_extracted_answers.py"

echo "QR=$QR"
echo "JSONL_ROOT=$JSONL_ROOT"
echo "OUT=$OUT"
echo "EXPORTER=$EXPORTER"

if [[ ! -f "$EXPORTER" ]]; then
  echo "ERROR: missing $EXPORTER" >&2
  echo "Checkout branch paper-p0-reanalysis in $QR first." >&2
  exit 2
fi

if [[ ! -d "$JSONL_ROOT" ]]; then
  echo "MISSING: $JSONL_ROOT"
  echo "Also check:"
  echo "  ls -ld $QR/outputs-hpc-campaign-2026-08-14"
  echo "  ls -ld $QR/outputs-hpc-breadth-gsm8k-2026-08-15"
  echo "  find $QR -maxdepth 3 -type d -name inference 2>/dev/null || true"
  exit 2
fi

echo "FOUND: $JSONL_ROOT"
echo "=== directory size ==="
du -sh "$JSONL_ROOT" || true
echo "=== jsonl count ==="
find "$JSONL_ROOT" -name '*.jsonl' | wc -l
echo "=== sample files ==="
# pipefail + head would otherwise abort on SIGPIPE
find "$JSONL_ROOT" -name '*.jsonl' | head -n 20 || true

echo ""
echo "=== peek first record keys (no GPU, no writes) ==="
python3 "$EXPORTER" --jsonl-root "$JSONL_ROOT" --peek

echo ""
echo "=== export sidecars (does not modify JSONLs) ==="
python3 "$EXPORTER" --jsonl-root "$JSONL_ROOT" --out "$OUT"

echo ""
echo "Export finished: $OUT"
echo "Campaign JSONLs were not modified."
echo "Rsync sidecars to MacBook; agreement must follow docs/ANSWER_NORMALIZATION.md"
echo "  (evaluator/math-verify is primary; do not invent a new judge)."
