# HPC block b08 — 2× A100 parallel, Qwen-1.5B BF16 + FP8 MATH-500.
# Future HPC-only lower-bound cells; submit after b01-b06 or when capacity allows.
# shellcheck disable=SC2034
HPC_BLOCK_ID="b08_qwen15b_bf16_fp8"
HPC_BLOCK_GPUS=2
HPC_BLOCK_EST_HOURS="12-24"
HPC_PARALLEL=true
HPC_BLOCK_CELLS=(
  "0:configs/cells/level_c_qwen15b_bf16_math500_seed0.json"
  "1:configs/cells/level_c_qwen15b_fp8_math500_seed0.json"
)
