# Legacy harness model configs (not the publication launcher)

These JSON files document Hugging Face IDs and revisions used by the **old local/HPC harness** (`configs/cells/*.json` → `scripts/run_inference.py`). They are **not** the frozen 56k QRM `inference.py` campaign.

Known mismatches versus the publication stack:

- `max_model_len` 40960 or 131072 (campaign: 32768)
- `gpu_memory_utilization` 0.95 in some files (campaign: 0.75)
- `kv_cache_dtype=fp8_e5m2` in some FP8 files (campaign: vLLM default; not passed)
- `enforce_eager=false` in some files (campaign: eager)

Publication runtime: [`../../results/reports/runtime_manifest.json`](../../results/reports/runtime_manifest.json).
See also [`../models/README.md`](../models/README.md).
