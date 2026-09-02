# `configs/models/` is not the frozen publication launcher

This directory is **intentionally empty of JSON**.

The 88-run / 56,408-completion campaign was launched through QRM `inference.py` (`scripts/hpc/qrm_parity/run_official_inference.sh` after `patches/qrm_hpc_compat.patch`). Effective settings are recorded in:

- [`../../results/reports/runtime_manifest.json`](../../results/reports/runtime_manifest.json)

Do **not** treat any file that used to live here as the A100 / vLLM 0.7.0 eager stack. Those historical harness configs used different `max_model_len`, `gpu_memory_utilization`, and sometimes `kv_cache_dtype` / `enforce_eager` defaults.

Historical JSON lives in [`../legacy_models/`](../legacy_models/).
