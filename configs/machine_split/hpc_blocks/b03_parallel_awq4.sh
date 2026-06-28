# HPC block b03 — 2× A100 parallel, ~12–24 h
# shellcheck disable=SC2034
HPC_BLOCK_ID="b03_parallel_awq4"
HPC_BLOCK_GPUS=2
HPC_BLOCK_EST_HOURS="12-24"
HPC_PARALLEL=true
HPC_BLOCK_CELLS=(
  "0:configs/cells/level_b_qwen7b_awq4_math500_seed0.json"
  "1:configs/cells/level_c_llama8b_awq4_math500_seed0.json"
)
