#!/usr/bin/env bash
# HPC 2× A100 (80 GB) publication blocks — journal protocol, ≤48 h SLURM jobs.
#
# All publication experiments run on HPC. Windows/5080 is retired for publication runs.
# Qwen-1.5B cells are future HPC-only work after model download/preflight.
#
# Usage:
#   bash scripts/hpc/run_hpc_2a100_publication.sh list
#   bash scripts/hpc/run_hpc_2a100_publication.sh b01_parallel_bf16_anchors
#   bash scripts/hpc/run_hpc_2a100_publication.sh b02_gpqa_fp8
#   bash scripts/hpc/submit_hpc_blocks.sh              # sbatch all ready blocks
#
# On PARAM Rudra after git pull:
#   export QR=/scratch/$USER/reasoning-compression-lab
#   cd $QR && bash scripts/hpc/run_hpc_2a100_publication.sh b01_parallel_bf16_anchors
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QR="${QR:-/scratch/$USER/reasoning-compression-lab}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/param_rudra_env.sh"
param_rudra_activate_conda

DATE_TAG="${QREASON_HPC_DATE:-$(date +%Y-%m-%d)}"
export QREASON_OUTPUT_ROOT="${QREASON_OUTPUT_ROOT:-$QR/outputs-hpc-2a100-main-${DATE_TAG}}"
RAW="$QREASON_OUTPUT_ROOT/raw"
SCORED="$QREASON_OUTPUT_ROOT/scored"
RESULTS="$QREASON_OUTPUT_ROOT/results"
LOGS="$QREASON_OUTPUT_ROOT/logs"
CHECKPOINTS="$QREASON_OUTPUT_ROOT/checkpoints"
METADATA="$QREASON_OUTPUT_ROOT/metadata"
BACKUP_ROOT="$QREASON_OUTPUT_ROOT/_backup"
mkdir -p "$RAW" "$SCORED" "$RESULTS" "$LOGS" "$CHECKPOINTS" "$METADATA" "$BACKUP_ROOT"

DECODING="${QREASON_DECODING:-configs/decoding/repro_qrm.yaml}"
BATCH_SIZE="${QREASON_BATCH_SIZE:-1}"
CHECKPOINT_EVERY="${QREASON_CHECKPOINT_EVERY:-10}"
export QR DATE_TAG BACKUP_ROOT DECODING BATCH_SIZE CHECKPOINT_EVERY

cd "$QR"

cell_id_from_cfg() {
  python -c "import json; print(json.load(open('$1'))['cell_id'])"
}

backup_archive() {
  python - <<'PY_BACKUP'
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["QR"])
from src.runners.checkpoint_utils import backup_mirror

backup_mirror(Path(os.environ["BACKUP_ROOT"]), Path(os.environ["QREASON_OUTPUT_ROOT"]))
PY_BACKUP
}

write_manifest_header() {
  local block_id="${1:-unknown}" block_file="${2:-}"
  export HPC_MANIFEST_BLOCK_ID="$block_id"
  export HPC_MANIFEST_BLOCK_FILE="$block_file"
  python - <<'PY_MANIFEST'
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ["QREASON_OUTPUT_ROOT"])
repo = Path(os.environ["QR"])
manifest_path = root / "manifest.json"
now = datetime.now(timezone.utc).isoformat()

def cmd(args):
    try:
        return subprocess.check_output(args, cwd=repo, text=True).strip()
    except Exception:
        return None

if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resumed_at"] = now
else:
    manifest = {
        "machine": "PARAM Rudra HPC 2x A100",
        "output_root": str(root),
        "publication_mode": True,
        "protocol": "hpc_repro_qrm",
        "started_at": now,
        "cells": [],
    }

manifest.update({
    "date": os.environ.get("DATE_TAG"),
    "block_id": os.environ.get("HPC_MANIFEST_BLOCK_ID"),
    "block_file": os.environ.get("HPC_MANIFEST_BLOCK_FILE") or None,
    "decoding_config": os.environ.get("DECODING"),
    "batch_size": int(os.environ.get("BATCH_SIZE", "1")),
    "checkpoint_every": int(os.environ.get("CHECKPOINT_EVERY", "10")),
    "git_commit": cmd(["git", "rev-parse", "HEAD"]),
    "git_branch": cmd(["git", "branch", "--show-current"]),
    "git_status_short": cmd(["git", "status", "--short"]),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
    "slurm_node_list": os.environ.get("SLURM_NODELIST"),
    "user": os.environ.get("USER"),
    "hostname": platform.node(),
    "updated_at": now,
})
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY_MANIFEST
}

write_cell_metadata() {
  local cell_id="$1" cell_cfg="$2" gpu_id="$3" status="$4" out="$5" summary="${6:-}"
  export HPC_META_CELL_ID="$cell_id"
  export HPC_META_CELL_CFG="$cell_cfg"
  export HPC_META_GPU_ID="$gpu_id"
  export HPC_META_STATUS="$status"
  export HPC_META_RAW="$out"
  export HPC_META_SUMMARY="$summary"
  python - <<'PY_METADATA'
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ["QREASON_OUTPUT_ROOT"])
repo = Path(os.environ["QR"])
cell_cfg = Path(os.environ["HPC_META_CELL_CFG"])
cell = json.loads((repo / cell_cfg).read_text(encoding="utf-8"))
model = json.loads((repo / cell["model_config"]).read_text(encoding="utf-8"))
task = json.loads((repo / cell["task_config"]).read_text(encoding="utf-8"))
decoding_path = Path(os.environ["DECODING"])
decoding_abs = repo / decoding_path
decoding_text = decoding_abs.read_text(encoding="utf-8") if decoding_abs.exists() else ""
raw_path = Path(os.environ["HPC_META_RAW"])
summary_env = os.environ.get("HPC_META_SUMMARY") or ""
summary_path = Path(summary_env) if summary_env else None
rows = 0
if raw_path.exists():
    rows = sum(1 for line in raw_path.open("r", encoding="utf-8") if line.strip())

try:
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
except Exception:
    git_commit = None

cell_id = os.environ["HPC_META_CELL_ID"]
payload = {
    "cell_id": cell_id,
    "status": os.environ["HPC_META_STATUS"],
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "gpu_id": os.environ["HPC_META_GPU_ID"],
    "rows_saved": rows,
    "raw": str(raw_path),
    "summary": str(summary_path) if summary_path else None,
    "cell_config_path": str(cell_cfg),
    "cell_config": cell,
    "model_config_path": cell["model_config"],
    "model_config": model,
    "task_config_path": cell["task_config"],
    "task_config": task,
    "decoding_config_path": os.environ["DECODING"],
    "decoding_config_text": decoding_text,
    "batch_size": int(os.environ.get("BATCH_SIZE", "1")),
    "checkpoint_every": int(os.environ.get("CHECKPOINT_EVERY", "10")),
    "git_commit": git_commit,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
    "slurm_node_list": os.environ.get("SLURM_NODELIST"),
}

metadata_dir = root / "metadata"
metadata_dir.mkdir(parents=True, exist_ok=True)
metadata_path = metadata_dir / f"{cell_id}.json"
metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

manifest_path = root / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"cells": []}
entry = {
    "cell_id": cell_id,
    "status": payload["status"],
    "raw": str(raw_path),
    "summary": str(summary_path) if summary_path else None,
    "metadata": str(metadata_path),
    "rows_saved": rows,
    "updated_at": payload["updated_at"],
}
cells = [c for c in manifest.get("cells", []) if c.get("cell_id") != cell_id]
cells.append(entry)
manifest["cells"] = cells
manifest["updated_at"] = payload["updated_at"]
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY_METADATA
}

run_one_cell() {
  local gpu_id="$1" cell_cfg="$2"
  local cell_id out log summary scored
  cell_id="$(cell_id_from_cfg "$cell_cfg")"
  out="$RAW/${cell_id}.jsonl"
  log="$LOGS/${cell_id}.log"
  scored="$SCORED/${cell_id}.jsonl"
  summary="$RESULTS/${cell_id}_summary.json"

  if [[ -f "$out" ]]; then
    local got want task
    got="$(wc -l < "$out" | tr -d ' ')"
    task="$(python -c "
import json
from pathlib import Path
cell = json.loads(Path('$cell_cfg').read_text())
task = json.loads(Path(cell['task_config']).read_text())
print(task['task_name'])
")"
    case "$task" in
      math500) want=500 ;;
      gsm8k) want=1319 ;;
      *) want=1 ;;
    esac
    if [[ "$got" -ge "$want" ]]; then
      echo "[gpu $gpu_id][skip] $cell_id complete ($got rows)"
      write_cell_metadata "$cell_id" "$cell_cfg" "$gpu_id" "completed" "$out" "$summary"
      backup_archive
      return 0
    fi
    echo "[gpu $gpu_id][resume] $cell_id — $got/$want rows"
  fi

  write_cell_metadata "$cell_id" "$cell_cfg" "$gpu_id" "in_progress" "$out" ""
  backup_archive
  echo "[gpu $gpu_id] === inference: $cell_id ==="
  (
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    python scripts/run_inference.py \
      --cell-config "$cell_cfg" \
      --decoding-config "$DECODING" \
      --batch-size "$BATCH_SIZE" \
      --checkpoint-every "$CHECKPOINT_EVERY" \
      --output "$out"
  ) 2>&1 | tee "$log"
  write_cell_metadata "$cell_id" "$cell_cfg" "$gpu_id" "inference_completed" "$out" ""
  backup_archive

  echo "[gpu $gpu_id] === score: $cell_id ==="
  rel_raw="${out#"$QR"/}"
  rel_scored="${scored#"$QR"/}"
  rel_summary="${summary#"$QR"/}"
  python scripts/score_run.py \
    --input "$rel_raw" \
    --output "$rel_scored" \
    --summary "$rel_summary" 2>&1 | tee -a "$log"
  write_cell_metadata "$cell_id" "$cell_cfg" "$gpu_id" "scored" "$out" "$summary"
  backup_archive
}

run_block() {
  local block_file="$1"
  # shellcheck disable=SC1090
  source "$block_file"
  echo "=== HPC block: $HPC_BLOCK_ID ==="
  echo "Archive: $QREASON_OUTPUT_ROOT"
  echo "GPUs: $HPC_BLOCK_GPUS | Est: ${HPC_BLOCK_EST_HOURS}h | Parallel: $HPC_PARALLEL"
  write_manifest_header "$HPC_BLOCK_ID" "$block_file"
  backup_archive

  if [[ "$HPC_PARALLEL" == "true" && "$HPC_BLOCK_GPUS" -ge 2 ]]; then
    pids=()
    for entry in "${HPC_BLOCK_CELLS[@]}"; do
      gpu="${entry%%:*}"
      cfg="${entry#*:}"
      run_one_cell "$gpu" "$cfg" &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do
      wait "$pid" || return 1
    done
  else
    for entry in "${HPC_BLOCK_CELLS[@]}"; do
      gpu="${entry%%:*}"
      cfg="${entry#*:}"
      run_one_cell "$gpu" "$cfg"
    done
  fi
  echo "=== Block $HPC_BLOCK_ID finished ==="
}

list_blocks() {
  echo "HPC publication blocks (SLURM --time=47:00:00 max):"
  echo ""
  echo "  b01_parallel_bf16_anchors  2×A100  ~12–24h  BF16 Qwen-7B + Llama-8B MATH"
  echo "  b02_parallel_fp8           2×A100  ~12–24h  FP8 Qwen-7B + Llama-8B MATH"
  echo "  b03_parallel_awq4          2×A100  ~12–24h  AWQ Qwen-7B + Llama-8B MATH"
  echo "  b04_parallel_gptq4         2×A100  ~12–24h  GPTQ-4 Qwen-7B + Llama-8B MATH"
  echo "  b05_single_gptq3           1×A100  ~12–20h  GPTQ-3 Qwen-7B MATH"
  echo "  b06_single_gsm8k           1×A100  ~20–40h  FP8 Qwen-7B GSM8K (n=1319)"
  echo "  b07_gpqa_fp8               1×A100  ~8–20h   GPQA (after HF gate)"
  echo "  b08_qwen15b_bf16_fp8       2×A100  ~12–24h  Qwen-1.5B BF16 + FP8 MATH"
  echo "  b09_qwen15b_awq4_gptq4     2×A100  ~12–24h  Qwen-1.5B AWQ-4 + GPTQ-4 MATH"
  echo ""
  echo "b08-b09 are future HPC-only lower-bound jobs; do not submit until current queue strategy allows."
}

BLOCK="${1:-list}"
BLOCK_DIR="$QR/configs/machine_split/hpc_blocks"

case "$BLOCK" in
  list|--list|-h|--help)
    list_blocks
    ;;
  b01|b01_parallel_bf16_anchors)
    run_block "$BLOCK_DIR/b01_parallel_bf16_anchors.sh"
    ;;
  b02|b02_parallel_fp8)
    run_block "$BLOCK_DIR/b02_parallel_fp8.sh"
    ;;
  b03|b03_parallel_awq4)
    run_block "$BLOCK_DIR/b03_parallel_awq4.sh"
    ;;
  b04|b04_parallel_gptq4)
    run_block "$BLOCK_DIR/b04_parallel_gptq4.sh"
    ;;
  b05|b05_single_gptq3)
    run_block "$BLOCK_DIR/b05_single_gptq3.sh"
    ;;
  b06|b06_single_gsm8k)
    run_block "$BLOCK_DIR/b06_single_gsm8k.sh"
    ;;
  b07|b07_gpqa_fp8|b02_gpqa_fp8)
    run_block "$BLOCK_DIR/b02_gpqa_fp8.sh"
    ;;
  b08|b08_qwen15b_bf16_fp8)
    run_block "$BLOCK_DIR/b08_qwen15b_bf16_fp8.sh"
    ;;
  b09|b09_qwen15b_awq4_gptq4)
    run_block "$BLOCK_DIR/b09_qwen15b_awq4_gptq4.sh"
    ;;
  *)
    echo "Unknown block: $BLOCK"
    list_blocks
    exit 1
    ;;
esac
