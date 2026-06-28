# RTX 5080 — ONLY cells that finish in ~≤24 h per cell (Qwen-1.5B).
# Everything 7B / 8B / GSM8K → HPC (scripts/hpc/run_hpc_2a100_publication.sh).
#
# Rule: if a cell is expected to exceed 1 day on 5080, it must not be listed here.
#
# shellcheck disable=SC2034
CELL_QUEUE=(
  configs/cells/level_c_qwen15b_bf16_math500_seed0.json
  configs/cells/level_c_qwen15b_fp8_math500_seed0.json
  configs/cells/level_c_qwen15b_awq4_math500_seed0.json
  configs/cells/level_c_qwen15b_gptq4_math500_seed0.json
)
