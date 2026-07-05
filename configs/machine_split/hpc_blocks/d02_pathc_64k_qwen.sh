# Path C budget diagnostic — Qwen-7B BF16 at 64k output cap, 50 problems, seed 42.
# shellcheck disable=SC2034
HPC_BLOCK_ID="d02_pathc_64k_qwen"
HPC_BLOCK_GPUS=1
HPC_BLOCK_EST_HOURS="8-16"
HPC_PARALLEL=false
HPC_BLOCK_CELLS=(
  "0:configs/cells/diag_qwen7b_bf16_math500_seed42_n50_64k.json"
)