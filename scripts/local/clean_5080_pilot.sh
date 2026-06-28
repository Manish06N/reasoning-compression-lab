#!/usr/bin/env bash
# Stop stale 5080 pilot jobs and reset inconsistent manifest/state for a clean restart.
#
# Usage:
#   bash scripts/local/clean_5080_pilot.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

DATE_TAG="${QREASON_5080_DATE:-2026-06-28}"
ARCHIVE="${QREASON_OUTPUT_ROOT:-$QR/outputs-win5080-pilot-${DATE_TAG}}"
LOGS="$ARCHIVE/logs"

echo "=== Stopping 5080 pilot processes ==="
pkill -f 'scripts/local/run_all_5080_phases.sh' 2>/dev/null || true
pkill -f 'scripts/run_inference.py' 2>/dev/null || true
pkill -f 'vllm.v1.engine.core' 2>/dev/null || true
pkill -f 'VLLM::EngineCore' 2>/dev/null || true
sleep 5

if pgrep -f 'run_all_5080|run_inference|vllm.v1.engine' >/dev/null 2>&1; then
  echo "WARN: some processes still running — sending SIGKILL"
  pkill -9 -f 'scripts/local/run_all_5080_phases.sh' 2>/dev/null || true
  pkill -9 -f 'scripts/run_inference.py' 2>/dev/null || true
  pkill -9 -f 'vllm.v1.engine.core' 2>/dev/null || true
  sleep 2
fi

echo "=== Cleaning manifest / checkpoints / partial raw ==="
python - <<PY
import json
from pathlib import Path

qr = Path("$QR")
archive = Path("$ARCHIVE")
raw = archive / "raw"
ckpt = archive / "checkpoints"
manifest_path = archive / "manifest.json"
state_path = archive / "state.json"
limit = 50

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
        want = 1 if cid.startswith("smoke_") else limit
        complete = rows >= want
        if status == "failed" and not complete:
            print(f"  drop manifest entry: {cid} (failed, {rows}/{want} rows)")
            continue
        if status == "in_progress" and rows == 0 and not raw_path.exists():
            print(f"  drop manifest entry: {cid} (stale in_progress, no raw file)")
            continue
        if status in ("in_progress", "failed") and complete:
            c["status"] = "completed" if cid.startswith("smoke_") else c["status"]
        kept.append(c)
    m["cells"] = kept
    manifest_path.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(f"  manifest: {len(kept)} cell entries kept")

# Remove orphan checkpoints with no matching complete raw
for p in ckpt.glob("*.json"):
    cid = p.stem
    raw_file = raw / f"{cid}.jsonl"
    rows = 0
    if raw_file.exists():
        rows = sum(1 for ln in raw_file.read_text(encoding="utf-8").splitlines() if ln.strip())
    prog = json.loads(p.read_text(encoding="utf-8"))
    if prog.get("rows_done", 0) == 0 and rows == 0:
        p.unlink()
        print(f"  removed orphan checkpoint: {p.name}")

# Remove empty / stale partial raw (0 bytes or corrupt) for non-smoke cells
for p in raw.glob("*.jsonl"):
    if p.name.startswith("smoke_"):
        continue
    rows = sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())
    if rows == 0:
        p.unlink()
        print(f"  removed empty raw: {p.name}")
    elif rows < limit:
        print(f"  keeping partial raw: {p.name} ({rows}/{limit} rows — will resume)")

# Drop tmp files
for p in raw.glob("*.tmp"):
    p.unlink()
    print(f"  removed tmp: {p.name}")

if state_path.exists():
    state_path.unlink()
    print("  reset state.json")
PY

mkdir -p "$LOGS"
SESSION="$(date -Is)"
echo "" >> "$LOGS/orchestrator.log"
echo "=== ${SESSION} clean restart (processes stopped, manifest pruned) ===" >> "$LOGS/orchestrator.log"

echo "=== GPU memory ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true
echo "Clean. Start with: bash scripts/local/start_5080_pilot.sh"
