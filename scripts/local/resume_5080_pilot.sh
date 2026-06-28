#!/usr/bin/env bash
# Resume the 5080 pilot grid after power cut / reboot / crash.
# Skips completed cells; continues partial JSONL from last checkpoint.
#
# Usage:
#   bash scripts/local/resume_5080_pilot.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_all_5080_phases.sh" --pilot --skip-download "$@"
