#!/usr/bin/env bash
# Publication-standard 5080 main grid (journal results on local RTX 5080).
#
# Protocol: repro_qrm.yaml, batch_size=1, full task sizes (MATH-500 n=500, etc.)
# Outputs → outputs-win5080-main-YYYY-MM-DD/
#
# Usage:
#   bash scripts/local/run_5080_main.sh
#   bash scripts/local/run_5080_main.sh --skip-download
# After power cut:
#   bash scripts/local/resume_5080_main.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_5080_publication.sh" "$@"
