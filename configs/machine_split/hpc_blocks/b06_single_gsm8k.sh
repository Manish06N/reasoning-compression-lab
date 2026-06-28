# HPC block b06 — 1× A100, ~20–40 h (1319 questions)
# shellcheck disable=SC2034
HPC_BLOCK_ID="b06_single_gsm8k"
HPC_BLOCK_GPUS=1
HPC_BLOCK_EST_HOURS="20-40"
HPC_PARALLEL=false
HPC_BLOCK_CELLS=(
  "0:configs/cells/level_b_qwen7b_fp8_gsm8k_seed0.json"
)
