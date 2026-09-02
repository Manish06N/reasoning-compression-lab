# `configs/cells/` is historical harness metadata

These JSON files (`level_a_*`, `level_b_*`, `diag_*`, `smoke_*`) belong to the **old local/HPC harness** (`scripts/run_inference.py`). They are **not** the frozen 88-run QRM `inference.py` campaign.

Reviewer traps in this directory:

- Paths may still mention `configs/models/` (that directory is now a warning README only).
- `level_b_qwen7b_gptq3_math500_seed0.json` refers to **GPTQ-3**, which was in the original design and **was not run**.
- `level_c_qwen15b_*` and `smoke_*` files are local-dev / 1.5B pipeline cells, not paper numbers.

Publication cells: compact JSON under `results/{math500,gsm8k,gpqa}/`. Effective stack: [`../../results/reports/runtime_manifest.json`](../../results/reports/runtime_manifest.json).
