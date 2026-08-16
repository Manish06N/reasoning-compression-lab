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

Optional matplotlib figures: `generate_paper_figures.py`. The submitted PDF draws figures in TikZ inside `paper/main.tex`; do not treat `paper_figures/` as the manuscript.

## Legacy

`legacy/` holds earlier scripts that can regenerate **wrong** paper numbers (mismatched pathology keys, gold-hit selective prediction, 200-item even-index token subset, external MATH-500 path). Do not run them for the manuscript.
