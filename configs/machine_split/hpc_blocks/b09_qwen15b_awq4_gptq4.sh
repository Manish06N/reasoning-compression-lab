# HPC block b09 — 2× A100 parallel, Qwen-1.5B AWQ-4 + GPTQ-4 MATH-500.
# Future HPC-only lower-bound cells; submit after b01-b06 or when capacity allows.
# shellcheck disable=SC2034
HPC_BLOCK_ID="b09_qwen15b_awq4_gptq4"
HPC_BLOCK_GPUS=2
HPC_BLOCK_EST_HOURS="12-24"
HPC_PARALLEL=true
HPC_BLOCK_CELLS=(
  "0:configs/cells/level_c_qwen15b_awq4_math500_seed0.json"
  "1:configs/cells/level_c_qwen15b_gptq4_math500_seed0.json"
)
