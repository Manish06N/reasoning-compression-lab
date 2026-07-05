# Path C diagnostic — strict QRM 32k, 50 MATH-500 problems, seed 42.
# GPU 0: Qwen-7B BF16 | GPU 1: Llama-8B BF16 (split 1-GPU jobs).
# shellcheck disable=SC2034
HPC_BLOCK_ID="d01_pathc_32k_diagnostic"
HPC_BLOCK_GPUS=2
HPC_BLOCK_EST_HOURS="6-12"
HPC_PARALLEL=true
HPC_BLOCK_CELLS=(
  "0:configs/cells/diag_qwen7b_bf16_math500_seed42_n50.json"
  "1:configs/cells/diag_llama8b_bf16_math500_seed42_n50.json"
)