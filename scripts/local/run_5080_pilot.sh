#!/usr/bin/env bash
# Fast 5080 pilot grid: n=50, pilot decoding (max_tokens 8192), vLLM batching.
# Outputs → outputs-win5080-pilot-YYYY-MM-DD/ (separate from full repro archive).
#
# Auto-resume: partial cells continue from last checkpoint; manifest preserved.
# Backups: _backup/latest/ (every cell) + _backup/snapshots/ (every 3 cells).
#
# Usage:
#   bash scripts/local/run_5080_pilot.sh
#   bash scripts/local/run_5080_pilot.sh --skip-download
# After power cut:
#   bash scripts/local/resume_5080_pilot.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_all_5080_phases.sh" --pilot "$@"
