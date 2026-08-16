# Paper 1 artifacts

**Preprint source:** `paper/main.tex` + `paper/references.bib` (figures are TikZ/pgfplots in the tex)  
**Compiled PDF:** `paper/main.pdf`  
**ArXiv upload zip:** `paper/arxiv_source.zip`

## What to upload to arXiv

Unzip `arxiv_source.zip` and upload:

- `main.tex`
- `references.bib`

Compile with `pdflatex` → `bibtex` → `pdflatex` → `pdflatex`. Figures are drawn by `pgfplots`; no separate PDFs are required.

## Result records (in the git repo)

| Path | Contents |
|------|----------|
| `results/README.md` | Scoreboard |
| `results/math500/` | 40 validation JSON files |
| `results/gsm8k/` | 24 validation JSON files |
| `results/gpqa/` | 24 validation JSON files |
| `results/reports/revision_reanalysis_report.json` | **Canonical** corrected pass@1 / pathology / token tables |
| `results/reports/modal_agreement_report.json` | Gold-free MATH-500 modal agreement |
| `results/recovered/math500_modal_inputs.jsonl` | Compact extracted answers (20,000 rows; no CoT / problem text) |
| `results/reports/runtime_manifest.json` | Effective 56k launch settings (not `configs/models/` defaults) |
| `results/reports/phase5_statistical_analysis_report.json` | Deprecation stub |
| `results/reports/multitask_benchmark_summary.json` | Deprecation stub |
| `results/reports/trace_audit_report.json` | Deprecation stub |
| `results/reports/selective_prediction_report.json` | Deprecation stub |

## How numbers were computed (read before citing)

- **Pass@1** is extractive match, averaged over seeds. Primary test: problem-clustered bootstrap of quantized − BF16. McNemar on maj@5 is secondary.
- **Loops** read `repetition_rows` (threshold: 20 consecutive identical words). **Cap hits** read `token_limit_hits`. **Near-cap** counts `completion_tokens >= 32500`.
- Compact per-cell JSON has **no** traces, token IDs, or `finish_reason`. Recovered MATH-500 answers are in `results/recovered/math500_modal_inputs.jsonl`. Do not report the old gold-hit 98.23% gate.
- Modal agreement uses unique-mode clustering with gold scoring only after the serve/abstain decision. Five-sample token cost $T_5$ sums all five seeds before abstention.
- **Cost-of-Pass** assumes $\$1.50$ per A100-hour and $65$ tok/s, shared across formats. It is a token ranking, not measured wall-clock.
- **Token inflation:** full-grid ratio of means over all seeds. The old 200-item mean-of-ratios subset is an estimator artifact and is not used in the paper.

## Reproduce the corrected tables

From a clean checkout (stdlib only; no `/scratch` or `outputs-hpc-*`):

```bash
python3 scripts/analysis/revision_reanalysis.py --check
python3 scripts/analysis/modal_agreement_analysis.py --check
```

Exits 0 only if the recomputed object matches `results/reports/revision_reanalysis_report.json`.

## Stack

Published 56k campaign: `requirements-qrm-paper-vllm070.lock` (`qrm-official`, vLLM 0.7.0, eager, A100-80GB). Effective launch settings: `results/reports/runtime_manifest.json`. The `qreason` file `requirements-hpc.txt` is **vLLM 0.8.5** and is labeled as legacy.

`configs/models/` is **not** the campaign launcher. HPC patches: `patches/qrm_hpc_compat.patch` and `patches/lighteval_local_dataset.patch`.
