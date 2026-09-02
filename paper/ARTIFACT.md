# Paper 1 artifacts

**Preprint source:** `paper/main.tex` + `paper/references.bib` (figures are TikZ/pgfplots in the tex)  
**Compiled PDF:** `paper/main.pdf` (22 pages, xelatex)
**ArXiv upload zip:** `paper/arxiv_source.zip` — rebuilt 2026-09-02 from current `main.tex` (80,846 B) + `references.bib` (7,942 B) + `main.bbl` (7,103 B). SHA256 `420d62c599c03bd7c313c1b2ebb89b2953c7d3c3a98cd2f7f07842b21643d346`. Do not upload a zip from before this rebuild.
**Submission tag:** `paper-v1.0-submission` (immutable snapshot of the canonical manuscript and analysis artifacts).

## What to upload to arXiv

Unzip `arxiv_source.zip` and upload:

- `main.tex`
- `references.bib`
- `main.bbl`

Compile with `xelatex` → `bibtex` → `xelatex` → `xelatex` (fallback: `pdflatex` if xelatex is unavailable). Figures are drawn by `pgfplots`; no separate PDFs are required. `main.bbl` is the frozen bibliography from the MacBook compile and should be uploaded with the source.

## Result records (in the git repo)

| Path | Contents |
|------|----------|
| `results/README.md` | Scoreboard |
| `results/math500/` | 40 validation JSON files |
| `results/gsm8k/` | 24 validation JSON files |
| `results/gpqa/` | 24 validation JSON files |
| `results/reports/revision_reanalysis_report.json` | **Canonical** corrected pass@1 / pathology / token tables |
| `results/reports/major_revision_tables.md` | Frozen paper-ready tables after independent recompute |
| `results/reports/modal_agreement_report.json` | Gold-free MATH-500 modal agreement |
| `results/recovered/math500_modal_inputs.jsonl` | Compact extracted answers (20,000 rows; no CoT / problem text) |
| `results/reports/measured_serving_confirmation/measured_serving_confirmation_report.json` | **Preferred** tok/s, latency, VRAM, GPU-sec/query, scenario $C_{\mathrm{pass}}$ |
| `results/measured_serving_confirmation/raw/` | 52 task-realistic + 8 microbenchmark confirmation JSON files |
| `results/reports/measured_serving/measured_serving_report.json` | First unconstrained timing (provenance only; not mixed with confirmation) |
| `results/measured_serving/raw/` | 48 task-realistic + 8 microbenchmark JSON files (superseded protocol) |
| `results/reports/runtime_manifest.json` | Effective 56k launch settings (not `configs/legacy_models/` defaults; `configs/models/` is not the launcher) |
| `results/reports/phase5_statistical_analysis_report.json` | Deprecation stub |
| `results/reports/multitask_benchmark_summary.json` | Deprecation stub |
| `results/reports/trace_audit_report.json` | Deprecation stub |
| `results/reports/selective_prediction_report.json` | Deprecation stub |

## How numbers were computed (read before citing)

- **Pass@1** is extractive match, averaged over seeds. Primary test: problem-clustered bootstrap of quantized − BF16 (two-sided bootstrap tail-area \(p\); percentile 95% CI). Primary multiplicity is Holm-6 within each benchmark; Holm-18 is a secondary sensitivity. McNemar on maj@5 is secondary.
- **Loops** read `repetition_rows` (threshold: 20 consecutive identical words). **Cap hits** read `token_limit_hits`. **Near-cap** counts `completion_tokens >= 32500`.
- Compact per-cell JSON has **no** traces, token IDs, or `finish_reason`. Recovered MATH-500 answers are in `results/recovered/math500_modal_inputs.jsonl`. Do not report the old gold-hit 98.23% gate.
- Modal agreement uses unique-mode clustering with gold scoring only after the serve/abstain decision. Five-sample token-cost proxy $T_5$ sums all five seeds before abstention.
- Campaign/extraction evaluator: **LightEval 0.8.0**. A throwaway MacBook LightEval 0.8.1 install was not used for paper numbers.
- **Aggregate hybrid Cost-of-Pass proxy (primary):** confirmation GPU-sec/query on `results/measured_serving_confirmation/` at $\$1.50$/A100-h (scenario) over campaign MATH-500 pass@1. Inspired by Erol et al., not their per-problem estimator. Report: `results/reports/measured_serving_confirmation/measured_serving_confirmation_report.json`. The first unconstrained timing in `results/measured_serving/` is provenance only. The old shared $65$ tok/s token ranking is a sensitivity only.
- **Token inflation:** full-grid ratio of means over all seeds. Clustered mismatch excess \(D\) (BF16-only mean \(\Delta\) minus Both-OK mean \(\Delta\)) is a diagnostic of correctness-conditioned mismatch asymmetry; it is not causal. The old 200-item mean-of-ratios subset is an estimator artifact and is not used in the paper.

## Reproduce the corrected tables

From a clean checkout (stdlib only; no `/scratch` or `outputs-hpc-*`):

```bash
python3 scripts/analysis/revision_reanalysis.py --check
python3 scripts/hpc/qrm_parity/benchmark_serving_confirmation.py --check
python3 scripts/analysis/emit_major_revision_tables.py --check
python3 scripts/analysis/measured_serving_confirmation_analysis.py --check
python3 scripts/analysis/modal_agreement_analysis.py --check-artifact
python3 scripts/analysis/item_level_descriptive_analysis.py --check
```

Expected PASS lines are listed in [`../REPRODUCE.md`](../REPRODUCE.md). `--check` fails on artifact drift and does not rewrite campaign results.

`revision_reanalysis.py --check` must match `results/reports/revision_reanalysis_report.json`. On MacBook, `modal_agreement_analysis.py --check-artifact` validates the compact artifact SHA and report internals (LightEval 0.8.0 re-extraction is HPC-only). `measured_serving_confirmation_analysis.py --check` recomputes aggregates from `results/measured_serving_confirmation/raw/`. `measured_serving_analysis.py --check` still recomputes the superseded first-run aggregates (provenance only).

Compact per-cell JSON has **no** full traces. Full GPU traces are not publicly released. Tables are reproducible; the complete GPU campaign is inspectable but not expected to be rerun by every reviewer.

## Stack

Published 56k campaign: `requirements-qrm-paper-vllm070.lock` (`qrm-official`, vLLM 0.7.0, eager, A100-80GB). Effective launch settings: `results/reports/runtime_manifest.json`. The `qreason` file `requirements-hpc.txt` is **vLLM 0.8.5** and is labeled as legacy.

`configs/models/` is **not** the campaign launcher (warning README only). Historical harness JSON: `configs/legacy_models/`. HPC patches: `patches/qrm_hpc_compat.patch` and `patches/lighteval_local_dataset.patch`.
