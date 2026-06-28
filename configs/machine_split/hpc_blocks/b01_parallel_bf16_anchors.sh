# HPC block b01 — 2× A100 parallel, ~12–24 h wall (under 48 h SLURM limit).
# GPU 0: Qwen-7B BF16 MATH-500 (Level A reproduction anchor)
# GPU 1: Llama-8B BF16 MATH-500
#
# Note: level_b_qwen7b_bf16 duplicates level_a — run level_a only on HPC.
#
# shellcheck disable=SC2034
HPC_BLOCK_ID="b01_parallel_bf16_anchors"
HPC_BLOCK_GPUS=2
HPC_BLOCK_EST_HOURS="12-24"
HPC_PARALLEL=true
HPC_BLOCK_CELLS=(
  "0:configs/cells/level_a_bf16_seed0.json"
  "1:configs/cells/level_c_llama8b_bf16_math500_seed0.json"
)
