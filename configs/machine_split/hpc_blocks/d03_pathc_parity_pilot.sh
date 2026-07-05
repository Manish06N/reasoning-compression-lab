# Path C parity pilot — Qwen-7B strict QRM with serving-stack fixes, n=10.
# Run after d01 diagnostic or in parallel if GPUs free.
# shellcheck disable=SC2034
HPC_BLOCK_ID="d03_pathc_parity_pilot"
HPC_BLOCK_GPUS=1
HPC_BLOCK_EST_HOURS="2-4"
HPC_PARALLEL=false
HPC_BLOCK_CELLS=(
  "0:configs/cells/diag_qwen7b_bf16_math500_seed42_n10_parity.json"
)