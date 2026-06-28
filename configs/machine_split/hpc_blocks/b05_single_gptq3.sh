# HPC block b05 — 1× A100, ~12–20 h (GPTQ-3 needs batch_size=1)
# shellcheck disable=SC2034
HPC_BLOCK_ID="b05_single_gptq3"
HPC_BLOCK_GPUS=1
HPC_BLOCK_EST_HOURS="12-20"
HPC_PARALLEL=false
HPC_BLOCK_CELLS=(
  "0:configs/cells/level_b_qwen7b_gptq3_math500_seed0.json"
)
