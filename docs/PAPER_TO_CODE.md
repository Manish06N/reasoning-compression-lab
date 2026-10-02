# Paper-to-code map

Where each table, figure and headline number of `paper/main.tex` comes from. Table and figure **labels** are used (the printed numbers follow the compiled PDF). All scripts are in `scripts/analysis/` unless noted and read only the released per-run records in `results/`.

Consistency checks: `python3 scripts/check_tex_tables.py --check` compares the manuscript's table cells with the frozen numbers, and `scripts/analysis/check_manuscript_numbers.py --check` tests that the frozen headline strings are present.

| Paper item (label) | Script | Output |
|---|---|---|
| `tab:headline_results`, `tab:multitask`, pass@1 per seed, loops / near-cap counts | `revision_reanalysis.py` | `results/reports/revision_reanalysis_report.json` |
| `tab:pass1`, `tab:gsm-contrasts`, `tab:gpqa-contrasts` (paired item-level bootstrap, Holm, TOST) | `revision_reanalysis.py`; tables via `emit_major_revision_tables.py` | `revision_reanalysis_report.json`, `major_revision_tables.md` |
| `tab:holm18` | `revision_reanalysis.py` | `revision_reanalysis_report.json` (`holm18_sensitivity`) |
| GPQA Qwen AWQ-4 borderline rerun ($B=10^6$, permutation, $t$-test) | `revision_sensitivities.py` | `revision_sensitivities.json` (`gpqa_borderline`) |
| `tab:tokens`, `tab:mismatch-excess` (token strata, conditionals, $D$) | `revision_reanalysis.py` | `revision_reanalysis_report.json` (`token_analysis`) |
| `tab:placebo` (BF16-vs-BF16 seed placebo) | `revision_sensitivities.py` | `revision_sensitivities.json` (`seed_placebo`) |
| `tab:levels`, difficulty join, per-item cap band, chat-template and GPQA-prompt audits | `q1_revision_analyses.py` (run in the frozen environment, see `audit_checks.py`) | `q1_revision_analyses.json`, `think_prefix_audit.json` |
| `tab:modal`, `tab:modal-paired`, `fig:modal` (answer agreement) | `modal_agreement_analysis.py` | `modal_agreement_report.json`, `modal_agreement_table.csv`; inputs `results/recovered/math500_modal_inputs.jsonl` |
| `tab:kcurve` ($k$-sample unanimity vs length rule) | `revision_sensitivities.py` | `revision_sensitivities.json` (`k_curve`) |
| `tab:length-abstain` (one-sample shortest-trace rule at the 5/5 coverage) | `manuscript_sensitivities.py` (stdout) and `revision_sensitivities.py` (`k_curve`, $k=5$) | printed by the script; $k=5$ entries in `revision_sensitivities.json` |
| `tab:serving-main`, `tab:serving-single`, `tab:serving-micro`, `tab:fp8-reps` | `measured_serving_confirmation_analysis.py` | `results/reports/measured_serving_confirmation/`; raw timing in `results/measured_serving_confirmation/raw/` |
| `tab:length-cost`, `tab:summary`, `fig:condB-scatter` (campaign-length seconds, rank bootstrap) | `manuscript_sensitivities.py` (stdout), inputs from the serving report and compact records | printed by the script; see also `measured_serving_confirmation_report.json` (`ranking_tables`) |
| `tab:economics`, `fig:tokens`, `fig:strata`, `fig:seed` | derived from the compact per-run JSON (mean tokens, pass@1, strata) | `revision_reanalysis_report.json` |
| Item-level descriptive claims (flips, failure vs success length) | `item_level_descriptive_analysis.py` | `item_level_descriptive_report.json` |
| Appendix "Additional sensitivities": `tab:twoway`, `tab:toklocation`, `tab:subsetsim` | `review_additions.py` | `review_additions.json` |
| Section "Failure modes and the length signal": `tab:failure`, `tab:auroc` | `trace_failure_analysis.py` | `trace_failure_analysis.json` |
| `tab:rope`, `tab:reflection`, `tab:adaptive` | `adaptive_reflection_rope.py` | `adaptive_reflection_rope.json` |
| Frozen-tokenizer prompt ranges, exact-cap test, tok/s aggregation, paired excess-over-placebo, Llama FP8 Condition A resampling | `audit_checks.py` (needs transformers 4.47.x) | `audit_checks.json` |
| `tab:checkpoints` (Hugging Face revisions), `tab:dtype` | `validate_runtime_manifest.py` | `results/reports/runtime_manifest.json` |
| `tab:qrm-compare`, `tab:related` (values from other papers) | none (literature values: QRM arXiv 2504.04823 v1 and v2, and the cited studies) | see `paper/references.bib` |
| Runtime configuration (vLLM 0.7.0, eager, dtype, kernels, seeds) | `validate_runtime_manifest.py` | `results/reports/runtime_manifest.json` |

## Notes

* `q1_revision_analyses.py` and `audit_checks.py` depend on the tokenizer implementation. Use the frozen environment (`transformers` 4.47.x); newer releases tokenize Llama text differently and give wrong prompt lengths.
* Analyses that sample (`review_additions.py`, `adaptive_reflection_rope.py`, `trace_failure_analysis.py`, `audit_checks.py`) use fixed seeds; the intervals are conditional on the five (MATH-500) or three (GSM8K, GPQA-Diamond) archived generation seeds.
* The GPU campaign scripts that produced `results/` live in `scripts/hpc/` and `src/`; they document how the data were generated but are not needed to check the analyses.
