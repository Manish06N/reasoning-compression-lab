#!/usr/bin/env bash
# RTX 5080 publication grid — journal protocol, all cells that fit in 16 GB VRAM.
#
# Protocol: repro_qrm.yaml, batch_size=1, full datasets (MATH-500 n=500, GSM8K n=1319)
# Archive:   outputs-win5080-main-YYYY-MM-DD/
#
# BF16 Qwen-7B / Llama-8B → HPC (run_hpc_2a100_publication.sh)
#
# Usage:
#   bash scripts/local/run_5080_publication.sh
#   bash scripts/local/run_5080_publication.sh --skip-download
#   bash scripts/local/start_5080_main.sh          # same grid, background
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QREASON_CELL_QUEUE="${QREASON_CELL_QUEUE:-$SCRIPT_DIR/../../configs/machine_split/5080_cells.sh}"
exec bash "$SCRIPT_DIR/run_all_5080_phases.sh" "$@"
