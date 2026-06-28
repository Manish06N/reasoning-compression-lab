#!/usr/bin/env bash
# Stop stale 5080 jobs and reset manifest/state for a clean restart.
#
# Usage:
#   bash scripts/local/clean_5080_run.sh                    # main archive (default)
#   bash scripts/local/clean_5080_run.sh pilot              # pilot archive
#   QREASON_OUTPUT_ROOT=... bash scripts/local/clean_5080_run.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

DATE_TAG="${QREASON_5080_DATE:-2026-06-28}"
MODE="${1:-main}"
if [[ "$MODE" == "pilot" ]]; then
  ARCHIVE="${QREASON_OUTPUT_ROOT:-$QR/outputs-win5080-pilot-${DATE_TAG}}"
else
  ARCHIVE="${QREASON_OUTPUT_ROOT:-$QR/outputs-win5080-main-${DATE_TAG}}"
fi
LOGS="$ARCHIVE/logs"

echo "=== Stopping 5080 processes (archive: $(basename "$ARCHIVE")) ==="
pkill -f 'scripts/local/run_all_5080_phases.sh' 2>/dev/null || true
pkill -f 'scripts/run_inference.py' 2>/dev/null || true
pkill -f 'vllm.v1.engine.core' 2>/dev/null || true
sleep 5

python - <<PY
import json
from pathlib import Path

qr = Path("$QR")
archive = Path("$ARCHIVE")
raw = archive / "raw"
ckpt = archive / "checkpoints"
manifest_path = archive / "manifest.json"
state_path = archive / "state.json"
limit = None
if manifest_path.exists():
    m0 = json.loads(manifest_path.read_text(encoding="utf-8"))
    limit = m0.get("limit")
default_want = 50 if limit else 500

if manifest_path.exists():
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    kept = []
    for c in m.get("cells", []):
        cid = c.get("cell_id", "")
        status = c.get("status", "")
        raw_path = Path(c.get("raw", "") or raw / f"{cid}.jsonl")
        if not raw_path.is_absolute():
            raw_path = qr / raw_path
        rows = 0
        if raw_path.exists():
            rows = sum(1 for ln in raw_path.read_text(encoding="utf-8").splitlines() if ln.strip())
        want = 1 if cid.startswith("smoke_") else (limit if limit else (1319 if "gsm8k" in cid else 500))
        complete = rows >= want
        if status == "failed" and not complete:
            print(f"  drop manifest entry: {cid} (failed, {rows}/{want})")
            continue
        if status == "in_progress" and rows == 0 and not raw_path.exists():
            print(f"  drop manifest entry: {cid} (stale in_progress)")
            continue
        kept.append(c)
    m["cells"] = kept
    manifest_path.write_text(json.dumps(m, indent=2), encoding="utf-8")

for p in ckpt.glob("*.json"):
    cid = p.stem
    raw_file = raw / f"{cid}.jsonl"
    rows = 0
    if raw_file.exists():
        rows = sum(1 for ln in raw_file.read_text(encoding="utf-8").splitlines() if ln.strip())
    if rows == 0:
        p.unlink(missing_ok=True)

if state_path.exists():
    state_path.unlink()

print(f"Clean complete: {archive}")
PY

mkdir -p "$LOGS"
echo "=== $(date -Is) clean restart ===" >> "$LOGS/orchestrator.log"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true
