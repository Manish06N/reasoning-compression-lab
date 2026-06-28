#!/usr/bin/env bash
# Mirror the current 5080 output archive to _backup/latest/ (+ optional snapshot).
#
# Usage:
#   source scripts/local/env.sh
#   export QREASON_OUTPUT_ROOT="$QR/outputs-win5080-pilot-2026-06-28"
#   bash scripts/local/backup_5080_archive.sh
#   bash scripts/local/backup_5080_archive.sh --snapshot
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

SNAPSHOT=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --snapshot) SNAPSHOT=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

ROOT="${QREASON_OUTPUT_ROOT:-}"
if [[ -z "$ROOT" ]]; then
  echo "Set QREASON_OUTPUT_ROOT (e.g. outputs-win5080-pilot-2026-06-28)"
  exit 1
fi

python - <<PY
import sys
sys.path.insert(0, "$QR")
from pathlib import Path
from src.runners.checkpoint_utils import backup_mirror, backup_snapshot

root = Path("$ROOT")
backup_root = root / "_backup"
if "$SNAPSHOT" == "true":
    snap = backup_snapshot(backup_root, root)
    print(f"Backed up → {backup_root / 'latest'}")
    print(f"Snapshot → {snap}")
else:
    backup_mirror(backup_root, root)
    print(f"Backed up → {backup_root / 'latest'}")
PY
