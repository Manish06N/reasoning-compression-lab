# Paper 1 artifacts

**Preprint source:** `paper/main.tex` + `paper/references.bib` (figures are TikZ/pgfplots in the tex)  
**Compiled PDF:** `paper/main.pdf` (12 pages)  
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
| `results/reports/revision_reanalysis_report.json` | **Canonical** corrected tables (pathology keys, clustered pass@1, token strata) |
| `results/reports/phase5_statistical_analysis_report.json` | Regenerated from the revision script |
| `results/reports/multitask_benchmark_summary.json` | MATH-500 / GSM8K / GPQA with real loop/near-cap counts |
| `results/reports/trace_audit_report.json` | Full-grid paired token analysis (not the old 200-item even-index subset) |
| `results/reports/selective_prediction_report.json` | Oracle gold-hit diagnostic, labeled as not deployable |

## How numbers were computed (read before citing)

- **Pass@1** is extractive match, averaged over seeds. Primary test: problem-clustered bootstrap of quantized − BF16. McNemar on maj@5 is secondary.
- **Loops** read `repetition_rows` (threshold: 20 consecutive identical words). **Cap hits** read `token_limit_hits`. **Near-cap** counts `completion_tokens >= 32500`.
- Compact JSON has **no** extracted answers, traces, token IDs, or `finish_reason`. Do not report a deployable agreement gate from this release.
- **Cost-of-Pass** assumes $\$1.50$ per A100-hour and $65$ tok/s, shared across formats. It is a token ranking, not measured wall-clock.
- **Token inflation:** full-grid ratio of means over all seeds. The old 200-item mean-of-ratios subset is an estimator artifact and is not used in the paper.

## Reproduce the corrected tables

From the repo root (stdlib only):

```bash
python3 scripts/analysis/revision_reanalysis.py
```

## Stack

Pinned `qrm-official`: vLLM 0.7.0, eager, A100-80GB. FP8 runs as Marlin W8A16 fallback. HPC patches: `patches/qrm_hpc_compat.patch` and `patches/lighteval_local_dataset.patch`. Files under `configs/models/` are **not** the campaign launcher.
