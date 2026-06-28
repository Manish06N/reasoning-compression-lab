#!/usr/bin/env bash
# Resume publication-standard 5080 main grid after power cut / reboot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_all_5080_phases.sh" "$@"
