#!/usr/bin/env bash
# Run all Paper 1 phases that fit on RTX 5080 (16 GB). No HPC.
# Outputs → outputs-win5080-* / (safe Windows archive on G: drive).
#
# Usage:
#   bash scripts/local/run_all_5080_phases.sh              # full datasets, repro decoding
#   bash scripts/local/run_all_5080_phases.sh --pilot      # n=50, pilot decoding, batched
#   bash scripts/local/run_all_5080_phases.sh --limit 10   # debug subset
#   bash scripts/local/run_all_5080_phases.sh --pilot --skip-download
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

LIMIT=""
PILOT=false
SKIP_DOWNLOAD=false
DECODING_CONFIG=""
MAX_MODEL_LEN=""
BATCH_SIZE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT="$2"; shift 2 ;;
    --pilot) PILOT=true; shift ;;
    --skip-download) SKIP_DOWNLOAD=true; shift ;;
    --decoding-config) DECODING_CONFIG="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

DATE_TAG="${QREASON_5080_DATE:-2026-06-28}"
if $PILOT; then
  LIMIT="${LIMIT:-50}"
  DECODING_CONFIG="${DECODING_CONFIG:-configs/decoding/pilot_5080.yaml}"
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
  BATCH_SIZE="${BATCH_SIZE:-}"
  export QREASON_OUTPUT_ROOT="${QREASON_OUTPUT_ROOT:-$QR/outputs-win5080-pilot-${DATE_TAG}}"
else
  # Publication-standard main grid (journal): QRM repro protocol, sequential inference.
  DECODING_CONFIG="${DECODING_CONFIG:-configs/decoding/repro_qrm.yaml}"
  BATCH_SIZE="${BATCH_SIZE:-1}"
  export QREASON_PUBLICATION_MODE=1
  export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
  export QREASON_OUTPUT_ROOT="${QREASON_OUTPUT_ROOT:-$QR/outputs-win5080-main-${DATE_TAG}}"
fi
RAW="$QREASON_OUTPUT_ROOT/raw"
SCORED="$QREASON_OUTPUT_ROOT/scored"
RESULTS="$QREASON_OUTPUT_ROOT/results"
LOGS="$QREASON_OUTPUT_ROOT/logs"
CHECKPOINTS="$QREASON_OUTPUT_ROOT/checkpoints"
BACKUP_ROOT="$QREASON_OUTPUT_ROOT/_backup"
SNAPSHOT_EVERY=3
CELLS_SINCE_SNAPSHOT=0
mkdir -p "$RAW" "$SCORED" "$RESULTS" "$LOGS" "$CHECKPOINTS" "$BACKUP_ROOT"

cleanup_vllm() {
  pkill -f 'vllm.v1.engine.core' 2>/dev/null || true
  pkill -f 'EngineCore pid' 2>/dev/null || true
  sleep 5
  nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || true
}

model_weights_ready() {
  local cell_cfg="$1"
  python - <<PY
import json, sys
from pathlib import Path
root = Path("$QR")
cell = json.loads((root / "$cell_cfg").read_text())
model = json.loads((root / cell["model_config"]).read_text())
import os
env = model.get("local_path_env", "")
default = model.get("local_path_default", "")
path = os.environ.get(env, default) if env else default
p = Path(path) if Path(path).is_absolute() else root / path
if not p.exists():
    sys.exit(1)
idx = p / "model.safetensors.index.json"
single = p / "model.safetensors"
if idx.exists() or single.exists():
    sys.exit(0)
if any(p.glob("*.safetensors")):
    sys.exit(0)
sys.exit(1)
PY
}

expected_rows() {
  local cell_cfg="$1"
  local task
  task="$(python -c "
import json, sys
from pathlib import Path
root = Path('$QR')
cell = json.loads((root / '$cell_cfg').read_text())
task = json.loads((root / cell['task_config']).read_text())
print(task['task_name'])
")"
  if [[ -n "$LIMIT" ]]; then
    echo "$LIMIT"
    return
  fi
  case "$task" in
    math500) echo 500 ;;
    gsm8k) echo 1319 ;;
    *) echo 1 ;;
  esac
}

cell_done() {
  local cell_id="$1"
  local cell_cfg="$2"
  local out="$RAW/${cell_id}.jsonl"
  [[ -f "$out" ]] || return 1
  local want got
  want="$(expected_rows "$cell_cfg")"
  got="$(wc -l < "$out" | tr -d ' ')"
  [[ "$got" -ge "$want" ]]
}

write_manifest_header() {
  python - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone

root = Path("$QREASON_OUTPUT_ROOT")
manifest_path = root / "manifest.json"
limit_val = "$LIMIT"
pilot = "$PILOT" == "true"
pub = not pilot
now = datetime.now(timezone.utc).isoformat()

if manifest_path.exists():
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    m["limit"] = int(limit_val) if limit_val else m.get("limit")
    m["pilot_mode"] = pilot
    m["publication_mode"] = pub
    m["protocol"] = "pilot_5080" if pilot else "repro_qrm"
    m["decoding_config"] = "$DECODING_CONFIG" or m.get("decoding_config")
    m["max_model_len_override"] = int("$MAX_MODEL_LEN") if "$MAX_MODEL_LEN" else m.get("max_model_len_override")
    m["batch_size"] = int("$BATCH_SIZE") if "$BATCH_SIZE" else m.get("batch_size")
    m["resumed_at"] = now
else:
    m = {
        "machine": "Windows WSL2 RTX 5080 16GB",
        "date": "$DATE_TAG",
        "output_root": str(root),
        "output_root_windows": "G:/ALL MY Projects/2026/03-paper1-experiments/" + root.name,
        "publication_mode": pub,
        "pilot_mode": pilot,
        "protocol": "pilot_5080" if pilot else "repro_qrm",
        "decoding_config": "$DECODING_CONFIG" or None,
        "max_model_len_override": int("$MAX_MODEL_LEN") if "$MAX_MODEL_LEN" else None,
        "batch_size": int("$BATCH_SIZE") if "$BATCH_SIZE" else None,
        "limit": int(limit_val) if limit_val else None,
        "started_at": now,
        "skipped_cells": [
            {"cell_id": "smoke_qwen7b_bf16_5080", "reason": "Qwen-7B BF16 ~14.3 GiB weights — no KV cache on 16 GB VRAM"},
            {"cell_id": "level_a_bf16_seed0", "reason": "Qwen-7B BF16 full MATH-500 — requires >16 GB VRAM (HPC when needed)"},
            {"cell_id": "level_b_qwen7b_bf16_math500_seed0", "reason": "Qwen-7B BF16 — requires >16 GB VRAM (HPC when needed)"},
            {"cell_id": "level_c_llama8b_bf16_math500_seed0", "reason": "Llama-8B BF16 — requires >16 GB VRAM (HPC when needed)"},
            {"cell_id": "level_c_qwen7b_fp8_gpqa_seed0", "reason": "GPQA-Diamond gated on Hugging Face"},
        ],
        "cells": [],
    }
manifest_path.write_text(json.dumps(m, indent=2), encoding="utf-8")
PY
}

backup_archive() {
  local do_snapshot="${1:-false}"
  python - <<PY
import sys
sys.path.insert(0, "$QR")
from pathlib import Path
from src.runners.checkpoint_utils import backup_mirror, backup_snapshot
root = Path("$QREASON_OUTPUT_ROOT")
backup_root = Path("$BACKUP_ROOT")
if "$do_snapshot" == "true":
    snap = backup_snapshot(backup_root, root)
    print(f"=== backup snapshot → {snap}")
else:
    backup_mirror(backup_root, root)
    print(f"=== backup mirror → {backup_root}/latest")
PY
}

scoring_current() {
  local raw="$1" scored="$2"
  [[ -f "$raw" && -f "$scored" ]] || return 1
  local raw_n scored_n
  raw_n="$(wc -l < "$raw" | tr -d ' ')"
  scored_n="$(wc -l < "$scored" | tr -d ' ')"
  [[ "$scored_n" -ge "$raw_n" && "$raw_n" -gt 0 ]]
}

infer_batch_size() {
  local cell_cfg="$1"
  if [[ -n "$BATCH_SIZE" ]]; then
    echo "$BATCH_SIZE"
    return
  fi
  python - <<PY
import json
from pathlib import Path
cell = json.loads((Path("$QR") / "$cell_cfg").read_text(encoding="utf-8"))
mc = cell.get("model_config", "").lower()
if "15b" in mc:
    print(4)
elif "gptq3" in mc:
    print(1)
else:
    print(2)
PY
}

update_manifest_cell() {
  local cell_id="$1" status="$2" raw_path="$3" summary_path="${4:-}"
  python - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone

p = Path("$QREASON_OUTPUT_ROOT/manifest.json")
m = json.loads(p.read_text(encoding="utf-8"))
entry = {
    "cell_id": "$cell_id",
    "status": "$status",
    "raw": "$raw_path",
    "summary": "$summary_path",
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
cells = [c for c in m.get("cells", []) if c.get("cell_id") != "$cell_id"]
cells.append(entry)
m["cells"] = cells
p.write_text(json.dumps(m, indent=2), encoding="utf-8")
PY
}

run_smoke() {
  local cell_cfg="$1"
  local cell_id out log
  cell_id="$(python -c "import json; print(json.load(open('$QR/$cell_cfg'))['cell_id'])")"
  out="$RAW/${cell_id}.jsonl"
  log="$LOGS/${cell_id}.log"
  local smoke_rows=0
  if [[ -f "$out" ]]; then
    smoke_rows="$(wc -l < "$out" | tr -d ' ')"
  fi
  if [[ "$smoke_rows" -ge 1 ]]; then
    echo "[skip] $cell_id (smoke complete: $smoke_rows row(s))"
    update_manifest_cell "$cell_id" "completed" "$out" ""
    return 0
  fi
  echo "=== smoke: $cell_id ==="
  local limit_args=(--limit 1 --max-tokens 64)
  if python scripts/smoke_test.py \
    --cell-config "$cell_cfg" \
    "${limit_args[@]}" \
    --output "$out" 2>&1 | tee "$log"; then
    update_manifest_cell "$cell_id" "completed" "$out" ""
    backup_archive false
    cleanup_vllm
    return 0
  fi
  update_manifest_cell "$cell_id" "failed" "$out" ""
  cleanup_vllm
  return 1
}

run_inference_cell() {
  local cell_cfg="$1"
  local cell_id out log summary scored
  cell_id="$(python -c "import json; print(json.load(open('$QR/$cell_cfg'))['cell_id'])")"
  out="$RAW/${cell_id}.jsonl"
  log="$LOGS/${cell_id}.log"
  scored="$SCORED/${cell_id}.jsonl"
  summary="$RESULTS/${cell_id}_summary.json"

  if cell_done "$cell_id" "$cell_cfg"; then
    echo "[skip] $cell_id (already complete: $out)"
  elif ! model_weights_ready "$cell_cfg"; then
    echo "[skip] $cell_id — model weights not downloaded yet"
    update_manifest_cell "$cell_id" "skipped_no_weights" "$out" ""
  else
    local partial=0
    if [[ -f "$out" ]]; then
      partial="$(wc -l < "$out" | tr -d ' ')"
      if [[ "$partial" -gt 0 ]]; then
        echo "[resume] $cell_id — $partial rows on disk, continuing inference"
      fi
    fi
    update_manifest_cell "$cell_id" "in_progress" "$out" ""
    echo "=== inference: $cell_id ==="
    local limit_arg=() decode_arg=() mlen_arg=() batch_arg=() ckpt_arg=()
    [[ -n "$LIMIT" ]] && limit_arg=(--limit "$LIMIT")
    [[ -n "$DECODING_CONFIG" ]] && decode_arg=(--decoding-config "$DECODING_CONFIG")
    [[ -n "$MAX_MODEL_LEN" ]] && mlen_arg=(--max-model-len "$MAX_MODEL_LEN")
    local bs
    bs="$(infer_batch_size "$cell_cfg")"
    batch_arg=(--batch-size "$bs")
    # Checkpoint: every batch (pilot) or every 10 rows (publication, batch=1).
    if $PILOT; then
      ckpt_arg=(--checkpoint-every "$bs")
    else
      ckpt_arg=(--checkpoint-every 10)
    fi
    if ! python scripts/run_inference.py \
      --cell-config "$cell_cfg" \
      "${limit_arg[@]}" \
      "${decode_arg[@]}" \
      "${mlen_arg[@]}" \
      "${batch_arg[@]}" \
      "${ckpt_arg[@]}" \
      --output "$out" 2>&1 | tee "$log"; then
      update_manifest_cell "$cell_id" "failed" "$out" ""
      cleanup_vllm
      return 1
    fi
    cleanup_vllm
  fi

  if scoring_current "$out" "$scored" && [[ -f "$summary" ]]; then
    echo "[skip] score: $cell_id (scored up to date)"
    update_manifest_cell "$cell_id" "scored" "$out" "$summary"
  else
    echo "=== score: $cell_id ==="
    rel_raw="${out#"$QR"/}"
    rel_scored="${scored#"$QR"/}"
    rel_summary="${summary#"$QR"/}"
    python scripts/score_run.py \
      --input "$rel_raw" \
      --output "$rel_scored" \
      --summary "$rel_summary" 2>&1 | tee -a "$log"
    update_manifest_cell "$cell_id" "scored" "$out" "$summary"
  fi

  CELLS_SINCE_SNAPSHOT=$((CELLS_SINCE_SNAPSHOT + 1))
  if (( CELLS_SINCE_SNAPSHOT >= SNAPSHOT_EVERY )); then
    backup_archive true
    CELLS_SINCE_SNAPSHOT=0
  else
    backup_archive false
  fi
}

echo "Output archive: $QREASON_OUTPUT_ROOT"
if $PILOT; then
  echo "Mode: PILOT (limit=${LIMIT}, decoding=${DECODING_CONFIG}, max_model_len=${MAX_MODEL_LEN})"
  echo "Windows path: G:\\ALL MY Projects\\2026\\03-paper1-experiments\\outputs-win5080-pilot-${DATE_TAG}"
else
  echo "Mode: PUBLICATION MAIN (repro_qrm, batch_size=${BATCH_SIZE}, full datasets)"
  echo "Windows path: G:\\ALL MY Projects\\2026\\03-paper1-experiments\\outputs-win5080-main-${DATE_TAG}"
fi
write_manifest_header

# Preserve earlier Phase 0 smoke if present (different filename).
if [[ -f "$QR/runs/raw/smoke_qwen15b_local.jsonl" && ! -f "$RAW/smoke_qwen15b_bf16.jsonl" ]]; then
  cp "$QR/runs/raw/smoke_qwen15b_local.jsonl" "$RAW/smoke_qwen15b_bf16.jsonl"
  echo "Copied prior smoke → $RAW/smoke_qwen15b_bf16.jsonl"
fi

echo ""
if $SKIP_DOWNLOAD; then
  echo "=== Skipping model download (--skip-download) ==="
else
  echo "=== Download 5080 quant models (skip if present) ==="
  bash "$SCRIPT_DIR/download_models.sh" 5080 || {
    echo "WARN: some downloads failed — will skip cells whose weights are missing"
  }
fi

# Sequential queue: load ONE model → run cell → score → unload vLLM → next.
run_smoke "configs/cells/smoke_qwen15b_bf16.json" || true
cleanup_vllm

if [[ -n "${QREASON_CELL_QUEUE:-}" && -f "${QREASON_CELL_QUEUE}" ]]; then
  # shellcheck source=/dev/null
  source "${QREASON_CELL_QUEUE}"
  echo "Cell queue: ${QREASON_CELL_QUEUE} (${#CELL_QUEUE[@]} cells)"
elif [[ -f "$QR/configs/machine_split/5080_cells.sh" ]]; then
  # shellcheck source=/dev/null
  source "$QR/configs/machine_split/5080_cells.sh"
  echo "Cell queue: configs/machine_split/5080_cells.sh (${#CELL_QUEUE[@]} cells)"
else
  CELL_QUEUE=(
    configs/cells/level_c_qwen15b_bf16_math500_seed0.json
    configs/cells/level_c_qwen15b_fp8_math500_seed0.json
    configs/cells/level_c_qwen15b_awq4_math500_seed0.json
    configs/cells/level_c_qwen15b_gptq4_math500_seed0.json
    configs/cells/level_a_gptq4_seed0.json
    configs/cells/level_b_qwen7b_fp8_math500_seed0.json
    configs/cells/level_b_qwen7b_awq4_math500_seed0.json
    configs/cells/level_b_qwen7b_gptq4_math500_seed0.json
    configs/cells/level_b_qwen7b_gptq3_math500_seed0.json
    configs/cells/level_b_qwen7b_fp8_gsm8k_seed0.json
    configs/cells/level_c_llama8b_fp8_math500_seed0.json
    configs/cells/level_c_llama8b_awq4_math500_seed0.json
    configs/cells/level_c_llama8b_gptq4_math500_seed0.json
  )
fi

for cfg in "${CELL_QUEUE[@]}"; do
  run_inference_cell "$cfg" || true
  cleanup_vllm
done

echo ""
echo "=== 5080 phase run finished ==="
backup_archive true
echo "Archive: $QREASON_OUTPUT_ROOT"
echo "  raw/      — inference JSONL"
echo "  scored/   — scored JSONL"
echo "  results/  — summary JSON"
echo "  logs/     — per-cell logs"
echo "  checkpoints/ — per-cell progress JSON"
echo "  _backup/  — auto mirror + snapshots (power-cut recovery)"
echo "  manifest.json, state.json"
