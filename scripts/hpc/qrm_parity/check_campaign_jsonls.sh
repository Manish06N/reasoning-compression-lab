#!/usr/bin/env bash
# Check whether the 56k campaign JSONLs still exist on PARAM Rudra, then export
# extracted answers if they do. CPU only. Run on the cluster, not on the MacBook.
#
#   ssh -p 4422 manishn_iitp@paramrudra.iitp.ac.in
#   export QR=/scratch/manishn_iitp/reasoning-compression-lab
#   bash $QR/scripts/hpc/qrm_parity/check_campaign_jsonls.sh
set -euo pipefail

QR="${QR:-/scratch/${USER}/reasoning-compression-lab}"
JSONL_ROOT="${1:-$QR/outputs-hpc-campaign-2026-08-14/inference}"
OUT="${2:-$QR/results/extracted_answers}"

echo "QR=$QR"
echo "JSONL_ROOT=$JSONL_ROOT"

if [[ ! -d "$JSONL_ROOT" ]]; then
  echo "MISSING: $JSONL_ROOT"
  echo "Also check:"
  echo "  ls -ld $QR/outputs-hpc-campaign-2026-08-14"
  echo "  ls -ld $QR/outputs-hpc-breadth-gsm8k-2026-08-15"
  echo "  find $QR -maxdepth 3 -type d -name inference 2>/dev/null"
  exit 2
fi

echo "FOUND: $JSONL_ROOT"
echo "=== directory size ==="
du -sh "$JSONL_ROOT" || true
echo "=== jsonl count ==="
find "$JSONL_ROOT" -name '*.jsonl' | wc -l
echo "=== sample files ==="
find "$JSONL_ROOT" -name '*.jsonl' | head -n 20

echo ""
echo "=== peek first record keys (no GPU) ==="
python3 - <<PY
import json, glob, os
root = os.environ.get("JSONL_ROOT", "${JSONL_ROOT}")
paths = sorted(glob.glob(root + "/**/*.jsonl", recursive=True))
print("n_jsonl", len(paths))
if not paths:
    raise SystemExit("no jsonl files")
with open(paths[0]) as f:
    row = json.loads(f.readline())
print("file", paths[0])
print("top_keys", sorted(row.keys()))
metrics = row.get("metrics") or {}
print("metrics_keys", sorted(metrics.keys()) if isinstance(metrics, dict) else type(metrics))
for k in ("text", "generated_text", "completion", "extracted_answer", "pred", "prediction"):
    if k in row or k in metrics:
        print("present:", k)
PY

mkdir -p "$OUT"
python3 "$QR/scripts/hpc/qrm_parity/export_extracted_answers.py" \
  --jsonl-root "$JSONL_ROOT" \
  --out "$OUT"

echo ""
echo "Export finished: $OUT"
echo "Rsync to MacBook, then implement agreement using docs/ANSWER_NORMALIZATION.md"
echo "  (evaluator/math-verify equivalence is primary; do not invent a new judge)."
