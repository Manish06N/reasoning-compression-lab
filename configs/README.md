# Configs: historical harness vs publication campaign

Do **not** treat this tree as a single serving stack. Most JSON/YAML here is the superseded local harness. **DO NOT USE** `models/`, `legacy_models/`, or `serving/` **FOR PAPER REPRODUCTION.**

```
configs/models/          historical only (warning README; no launcher JSON)
configs/legacy_models/   historical harness JSON (wrong max_model_len / KV defaults)
configs/serving/         historical vLLM 0.8.5 / SGLang / llama.cpp sketches
configs/cells/           historical cell lists (includes unrun GPTQ-3 and 1.5B smoke)
configs/publication/     pointers to the frozen Paper 1 campaign (not a second launcher)
configs/tasks/           frozen dataset IDs and SHAs (MATH-500, GSM8K, GPQA-Diamond)
```

| Path | Role |
|------|------|
| `publication/` | **Start here for Paper 1.** Pointers to `runtime_manifest.json`, task SHAs, vLLM 0.7.0 lockfile |
| `tasks/*.json` | Dataset IDs and **frozen SHAs** |
| `decoding/` | Historical harness decoding YAML |
| `cells/` | Historical harness cell lists |
| `campaign_cells.json`, `campaign_cells_gsm8k.json`, `campaign_cells_gpqa.json` | Historical scratch-path job lists, **not** the QRM launcher |
| `machine_split/` | Historical SLURM block scripts |
| `models/` | Warning only — **not** the 56k launcher |
| `legacy_models/` | Historical harness JSON |
| `serving/` | Historical vLLM **0.8.5** YAML — **not** Paper 1 |

The published campaign was launched with QRM `inference.py` (`scripts/hpc/qrm_parity/run_official_inference.sh`) after `patches/qrm_hpc_compat.patch`. Effective flags: [`../results/reports/runtime_manifest.json`](../results/reports/runtime_manifest.json). Do not delete historical files; they document the superseded harness.
