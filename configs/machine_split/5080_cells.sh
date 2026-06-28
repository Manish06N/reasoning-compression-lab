# Publication cells for RTX 5080 (16 GB VRAM).
# BF16 Qwen-7B / Llama-8B → HPC (scripts/hpc/run_hpc_2a100_publication.sh).
#
# shellcheck disable=SC2034
CELL_QUEUE=(
  configs/cells/level_c_qwen15b_bf16_math500_seed0.json
  configs/cells/level_c_qwen15b_fp8_math500_seed0.json
  configs/cells/level_c_qwen15b_awq4_math500_seed0.json
  configs/cells/level_c_qwen15b_gptq4_math500_seed0.json
  configs/cells/level_a_gptq4_seed0.json
  configs/cells/level_b_qwen7b_fp8_math500_seed0.json
  configs/cells/level_b_qwen7b_awq4_math500_seed0.json
  configs/cells/level_b_qwen7b_gptq4_math500_seed0.json
  configs/cells/level_b_qwen7b_gptq3_math500_seed0.json
  configs/cells/level_b_qwen7b_fp8_gsm8k_seed0.json
  configs/cells/level_c_llama8b_fp8_math500_seed0.json
  configs/cells/level_c_llama8b_awq4_math500_seed0.json
  configs/cells/level_c_llama8b_gptq4_math500_seed0.json
)
