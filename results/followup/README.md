# Follow-up experiments (October 2026)

Per-item records, checks and analysis code for the experiments added after the 88-run campaign. Draft manuscript: `paper/v2_followups/`.

| Id | What | Cells |
|---|---|---|
| E1 | Llama-8B AWQ-4 quantized by the authors (AutoAWQ 0.2.9, 4-bit, group 128, `zero_point=True`), MATH-500 and GSM8K | 5 + 3 |
| E2 | BF16 weights run at `float16` (both families), MATH-500 | 10 |
| E3 | BF16 sampled at T=0.8 instead of 0.6 (both families), MATH-500 | 10 |
| X1 | Five held-out BF16 seeds (47-51), both families, MATH-500 | 10 |

Not included (still running or not run): E1c (forced AWQ GEMM kernel), X3 (batch-limit placebo), X5 (greedy configuration placebo), X2 (AIME placebo), E4 (third model size). The verdict rules for E1c, X3 and X5 are in `PREREG_E1c_X3_X5.md`, committed before those cells existed.

## Layout
- `revision2/compact/`, `revision3/compact/` - compact per-item records, same schema as `results/math500/*.json` (`details[].row`, `completion_tokens`, `extractive_match`) plus `finish_reason` where the sidecar survived.
- `revision2/engine_settings/`, `manifests/`, `cell_status.tsv`, `status_supersessions.tsv`, `VERIFIER_EXPECTATIONS.md`, `LOG.md` - what each cell ran with, which checks it passed, the expected values (written before the reruns), and the incident log.
- `ENGINE_SETTINGS_SUMMARY.tsv`, `COMPACT_FINISH_REASON_COVERAGE.tsv`, `EXPORT_NOTES.md` - audit summaries and known gaps. Read `EXPORT_NOTES.md`: the MATH-500 cells of E1 for seeds 42-44 lost their raw output, stop reasons and engine dump to an output-directory collision (the per-item records and status rows survive).
- `provenance/` - script hashes and versions, the MATH-500 item table (row to dataset item, subject, level), rows whose engine and stored token counts differ.
- `reports/` - outputs of the analysis scripts.
- `revision2/_unverified_run1/` - metadata of the first, aborted attempt; excluded from every number.

## Reproduce (CPU, numpy only)
```bash
python scripts/analysis/followup/revision2_results.py      # E1, E2, E3, X1, per-level, 10-seed placebo -> results/followup/reports/
python scripts/analysis/followup/gen_followup_tables.py    # paper tables
python scripts/analysis/followup/gen_lenvar.py             # within-item length spread table
python scripts/analysis/followup/gen_fig_placebo.py        # Figure 1
python scripts/analysis/followup/test_seed_placebo.py      # tests of the standalone estimator
```
`scripts/analysis/followup/seed_placebo.py` is a standalone implementation of the placebo estimator; `prop1_checks.py` and `flips_and_intervals.py` are the earlier CPU checks of Proposition 1 and of the flip and interval analyses.

## Notes
- All follow-up prompts reach the engine with two beginning-of-sequence tokens (the prompt string starts with one and the pipeline adds another); the campaign shares this. Described in the manuscript appendix.
- Full completion text and the HPC job-accounting tables are not released.
