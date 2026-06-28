# HPC block b04 — 2× A100 parallel, ~12–24 h
# GPU 0: Qwen-7B GPTQ-4 (Level A anchor)
# GPU 1: Llama-8B GPTQ-4
# shellcheck disable=SC2034
HPC_BLOCK_ID="b04_parallel_gptq4"
HPC_BLOCK_GPUS=2
HPC_BLOCK_EST_HOURS="12-24"
HPC_PARALLEL=true
HPC_BLOCK_CELLS=(
  "0:configs/cells/level_a_gptq4_seed0.json"
  "1:configs/cells/level_c_llama8b_gptq4_math500_seed0.json"
)
