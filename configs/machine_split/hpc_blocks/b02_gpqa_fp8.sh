# HPC block b02 — 1× A100, ~8–20 h (when GPQA access granted).
# shellcheck disable=SC2034
HPC_BLOCK_ID="b02_gpqa_fp8"
HPC_BLOCK_GPUS=1
HPC_BLOCK_EST_HOURS="8-20"
HPC_PARALLEL=false
HPC_BLOCK_CELLS=(
  "0:configs/cells/level_c_qwen7b_fp8_gpqa_seed0.json"
)
