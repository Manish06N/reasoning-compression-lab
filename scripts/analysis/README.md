# Analysis scripts

## Canonical pipeline

```text
results/{math500,gsm8k,gpqa}/*.json
        ↓
python3 scripts/analysis/revision_reanalysis.py
        ↓
results/reports/revision_reanalysis_report.json
        ↓
paper/main.tex
```

Stdlib only. Reads **only** `results/` in this checkout. Does not open `outputs-hpc-campaign-*`, `/scratch/`, or `external/`.

```bash
python3 scripts/analysis/revision_reanalysis.py          # write canonical JSON
python3 scripts/analysis/revision_reanalysis.py --check  # CI / clean-clone drift test
```

`--check` exits nonzero if the recomputed object differs from the checked-in report (float abs_tol `1e-9`).

Paper-ready tables (independent recompute, including mismatch excess \(D\) and Holm-18):

```bash
python3 scripts/analysis/emit_major_revision_tables.py          # write major_revision_tables.md
python3 scripts/analysis/emit_major_revision_tables.py --check  # fail if generated markdown differs from frozen files
python3 scripts/hpc/qrm_parity/benchmark_serving_confirmation.py --check
# expected: OK: confirmation raw artifacts present (60 required files)
```

`--check` does not rewrite frozen JSON. Primary inference remains Holm-6 within each benchmark. Holm-18 is a secondary sensitivity only. Mismatch excess \(D\) is clustered on the same item-resample stream as the stratum CIs; it is not a causal claim.

CPU-only item-level descriptive summaries (flips, length vs correctness, GPQA row indices; no causality):

```bash
python3 scripts/analysis/item_level_descriptive_analysis.py --check
```

Optional matplotlib figures: `generate_paper_figures.py`. The submitted PDF draws figures in TikZ inside `paper/main.tex`; do not treat `paper_figures/` as the manuscript. The 2×2 length figure title is “Mismatch-associated lengthening vs jointly-correct pairs,” not “architecture dependent.”

## Legacy

`legacy/` holds earlier scripts that can regenerate **wrong** paper numbers (mismatched pathology keys, gold-hit selective prediction, 200-item even-index token subset, external MATH-500 path). Do not run them for the manuscript.
