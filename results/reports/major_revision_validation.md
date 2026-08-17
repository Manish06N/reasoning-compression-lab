# Major revision validation

CPU-only independent recompute via `scripts/analysis/emit_major_revision_tables.py`.
Does not replace `revision_reanalysis.py --check`.

- Independent 2×2 n/mean, Lian mean, and mismatch-excess point estimate: **OK**
- Independent GPQA Qwen AWQ clustered contrast: **OK**
- Independent hybrid $C_{pass}$ points: **OK**
- Holm-18 adjusted p / decisions vs recomputed Holm: **OK**

## Holm-18 status changes

- GPQA-Diamond Qwen-7B BF16 vs AWQ-4: within-benchmark significant → global-18 not significant (p=0.0068, holm18_p=0.1088)

