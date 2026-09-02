# Reproduce Paper 1 tables (CPU)

This repository is a **stack-pinned measurement study**. Frozen numbers come from the completed 88-run GPU campaign. Do not launch new GPU jobs to “improve” results.

The contribution is: under a fixed serving stack, public quantized reasoning checkpoints can change rank depending on what is measured. It is **not** a claim that one quantization method is best.

## What you can reproduce on a laptop

Tables and reports are recomputed from released compact JSON. No GPU.

```bash
python3 scripts/analysis/revision_reanalysis.py --check
# expected: OK: recomputed report matches .../revision_reanalysis_report.json

python3 scripts/hpc/qrm_parity/benchmark_serving_confirmation.py --check
# expected: OK: confirmation raw artifacts present (60 required files)

python3 scripts/analysis/emit_major_revision_tables.py --check
# expected: OK: generated tables match .../major_revision_tables.md and .../major_revision_validation.md
```

Additional CPU checks used in CI:

```bash
python3 scripts/analysis/measured_serving_confirmation_analysis.py --check
# expected: OK: recomputed confirmation report matches ...measured_serving_confirmation_report.json (...)

python3 scripts/analysis/modal_agreement_analysis.py --check-artifact
# expected: OK: compact artifact SHA256, 20,000/4,000 structure, T5 accounting, and report internals match.

python3 scripts/analysis/item_level_descriptive_analysis.py --check
# expected: OK: item-level descriptive report matches .../item_level_descriptive_report.json
```

`--check` compares generated output to frozen files and **fails on drift**. It does not rewrite campaign results.

## What requires the original GPU campaign

Tables are reproducible on a laptop from compact artifacts.

The complete GPU campaign is inspectable from the launcher, runtime manifest, and released compact JSON. It is **not** expected to be rerun by every reviewer.

Re-running the 88 accuracy cells or the confirmation serving jobs needs:

- NVIDIA A100-80GB
- vLLM 0.7.0 eager (`requirements-qrm-paper-vllm070.lock`)
- public checkpoint weights at the revisions in `results/reports/runtime_manifest.json`
- MATH-500 / GSM8K / GPQA-Diamond at the dataset SHAs in that manifest
- original campaign traces (full chain-of-thought text is **not** publicly released; compact per-cell JSON and confirmation timing JSON **are** under `results/`)

Compact campaign JSON omits `finish_reason` and output token IDs.

FP8 checkpoints were executed as **Marlin W8A16 on A100**, not native W8A8.

## Frozen sources of truth

| Artifact | Role |
|----------|------|
| `results/{math500,gsm8k,gpqa}/*.json` | Compact per-cell records |
| `results/reports/revision_reanalysis_report.json` | Canonical pass@1 / length / Holm |
| `results/reports/major_revision_tables.md` | Frozen paper tables |
| `results/reports/runtime_manifest.json` | Effective launch stack + dataset SHAs + LightEval 0.8.0 |
| `results/measured_serving_confirmation/raw/` | Confirmation GPU-seconds |
| `configs/models/` | **Not** the publication launcher (see README there) |

Manuscript: `paper/main.tex`.
