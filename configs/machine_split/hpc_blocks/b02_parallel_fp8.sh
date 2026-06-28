# HPC block b02 — 2× A100 parallel, ~12–24 h
# GPU 0: Qwen-7B FP8 × MATH-500
# GPU 1: Llama-8B FP8 × MATH-500
# shellcheck disable=SC2034
HPC_BLOCK_ID="b02_parallel_fp8"
HPC_BLOCK_GPUS=2
HPC_BLOCK_EST_HOURS="12-24"
HPC_PARALLEL=true
HPC_BLOCK_CELLS=(
  "0:configs/cells/level_b_qwen7b_fp8_math500_seed0.json"
  "1:configs/cells/level_c_llama8b_fp8_math500_seed0.json"
)
