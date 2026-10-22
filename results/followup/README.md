# Follow-up experiments (October 2026)

Per-item records, checks and analysis code for the experiments added after the 88-run campaign. Draft manuscript: `paper/v2_followups/`.

| Id | What | Cells |
|---|---|---|
| E1 | Llama-8B AWQ-4 quantized by the authors (AutoAWQ 0.2.9, 4-bit, group 128, `zero_point=True`), MATH-500 and GSM8K | 5 + 3 |
| E2 | BF16 weights run at `float16` (both families), MATH-500 | 10 |
| E3 | BF16 sampled at T=0.8 instead of 0.6 (both families), MATH-500 | 10 |
| E1c | The E1 weights rerun on the forced AWQ GEMM kernel, MATH-500 | 5 |
| X1 | Five held-out BF16 seeds (47-51), both families, MATH-500 | 10 |
| X3 | Qwen BF16 with the scheduler limit `max_num_seqs` 64 instead of 256, MATH-500 | 5 |
| P2 | Companion study: Qwen-7B GPTQ-4 and AWQ-4 quantized by the authors, 3 calibration draws x seeds 42-44, on MATH-500 and GSM8K (36 cells), and a float16 control for GPTQ-4 (18 cells) | 54 |
| P4 | Companion study: decoding grid, temperature {0.2, 0.6, 1.0} x token budget {8192, 32768}, BF16 and AWQ-4, 300 MATH-500 items, seeds 42-44 (pilot-only in the repository's wording) | 33 |

Not included (still queued or not run): X5 (greedy configuration placebo), X2 (AIME placebo), E4 (third model size), the P2 GSM8K and Llama cells. The verdict rules for E1c, X3 and X5 are in `PREREG_E1c_X3_X5.md`, committed before those cells finished; those for X2 and E4 are in `PREREG_X2_E4_as_received_2026-10-18.md`.

The P2 and P4 runs stored exact-match scores only. Their stored completions were rescored with the campaign's scoring functions (LightEval 0.8.0 metric objects), validated on 500/500 campaign MATH-500 rows and 1,319/1,319 E1 GSM8K rows (`p2p4/tables/scorer_validation.txt`). Rows are matched to the campaign's by item id (`provenance/MATH500_item_table.tsv`) or, for GSM8K, by dataset index (`provenance/GSM8K_item_table.tsv`; the two orders differ and must not be joined on row number).

Prompt caveat: the P2 cells and the AWQ-4 cells of the P4 grid have exactly one prompt token fewer than the campaign pipeline on every MATH-500 item (single instead of doubled beginning-of-sequence token, checked against the E2 cells); the P4 BF16 cells match the campaign. The effect of that token is untested.

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
python scripts/analysis/followup/e1c_x3_results.py         # E1c and X3 under the pre-registered rules
python scripts/analysis/followup/p2_qwen_results.py        # self-quantized Qwen checkpoints (P2 cells), MATH-500
python scripts/analysis/followup/p2_qwen_gsm8k_results.py  # the same, GSM8K
python scripts/analysis/followup/p4_decoding_results.py    # decoding-grid controls (P4 cells)
python scripts/analysis/followup/gen_followup_tables.py    # paper tables
python scripts/analysis/followup/gen_p2p4_tables.py        # P2 and P4 tables
python scripts/analysis/followup/gen_lenvar.py             # within-item length spread table
python scripts/analysis/followup/gen_fig_placebo.py        # Figure 1
python scripts/analysis/followup/test_seed_placebo.py      # tests of the standalone estimator
```
`scripts/analysis/followup/seed_placebo.py` is a standalone implementation of the placebo estimator; `prop1_checks.py` and `flips_and_intervals.py` are the earlier CPU checks of Proposition 1 and of the flip and interval analyses.

## Notes
- All follow-up prompts reach the engine with two beginning-of-sequence tokens (the prompt string starts with one and the pipeline adds another); the campaign shares this. Described in the manuscript appendix.
- Full completion text and the HPC job-accounting tables are not released.
