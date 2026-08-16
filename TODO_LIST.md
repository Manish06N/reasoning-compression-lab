# TODO List & Experiment Roadmap — Publication Readiness
**Project:** Reasoning Compression Lab (`reasoning-compression-lab`)  
**Target:** Paper 1 Publication Readiness (Quantization, Reliability, and Cost Frontier for Reasoning Models)  
**Status:** **88-cell campaign complete. P0 reanalysis on branch `paper-p0-reanalysis`. Do not cite 0/0 pathologies or the 98.23% safety gate.**  
**Controlling Reference:** [docs/PUBLICATION_READINESS.md](docs/PUBLICATION_READINESS.md) · [docs/plans/2026-08-14-publication-recovery.md](docs/plans/2026-08-14-publication-recovery.md) · [AGENTS.md](AGENTS.md)

---

## 1. Executive Summary & Gating Decision

* **Headline MATH-500 ($n=500$, Seeds 42–46, 40 Cells):** ✅ 100% Completed, validated, and backed up.
* **Breadth Benchmark 1 — GSM8K ($n=1,319$, Seeds 42–44, 24 Cells):** ✅ 100% Completed, validated, and backed up.
* **Breadth Benchmark 2 — GPQA-Diamond ($n=198$, Seeds 42–44, 24 Cells):** ✅ 100% Completed, validated, and backed up.
* **Manuscript Status:** Canonical source [`paper/main.tex`](paper/main.tex) compiled to [`paper/main.pdf`](paper/main.pdf) (12 pages, 8 tables, 4 figures). Markdown mirror: [`paper/main.md`](paper/main.md). ArXiv zip: [`paper/arxiv_source.zip`](paper/arxiv_source.zip).

---

## 2. Completed Experiments & Verified Campaign Matrices

Pathology on the **full 88-cell grid** (not this table's old 0/0 columns): **25 loops**, **0 exact cap hits**, **209 near-cap**. See `results/README.md`.

### A. MATH-500 Headline Confirmatory Matrix ($n=500$, 5 Seeds)
| Model | Quant Format | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 | Mean ± Std | Truncations | Repetition Loops | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen-7B** | **BF16** | 94.4% | 94.0% | 93.8% | 94.6% | 93.2% | **94.00% ± 0.55%** | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **FP8** | 94.4% | 95.2% | 94.8% | 92.6% | 95.0% | **94.40% ± 1.05%** | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **AWQ-4** | 92.4% | 92.8% | 93.2% | 93.0% | 94.2% | **93.12% ± 0.67%** | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **GPTQ-4** | 93.8% | 92.6% | 93.4% | 94.6% | 93.0% | **93.48% ± 0.77%** | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **BF16** | 89.0% | 88.4% | 90.2% | 89.8% | 88.8% | **89.24% ± 0.74%** | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **FP8** | 89.0% | 89.6% | 88.6% | 89.2% | 91.2% | **89.52% ± 1.01%** | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **AWQ-4** | 84.4% | 84.8% | 89.2% | 87.4% | 86.6% | **86.48% ± 1.96%** | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **GPTQ-4** | 88.0% | 89.6% | 86.8% | 89.4% | 90.8% | **88.92% ± 1.55%** | 0 | 0 | ✅ COMPLETED |

### B. GSM8K Breadth Matrix ($n=1,319$, 3 Seeds: 42, 43, 44)
| Model | Quant Format | Seed 42 | Seed 43 | Seed 44 | Mean ± Std | Mean Tokens | Truncations | Repetition Loops | Status |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen-7B** | **BF16** | 91.1% | 91.6% | 91.1% | **91.26% ± 0.29%** | 1695.9 | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **FP8** | 91.3% | 91.5% | 91.2% | **91.33% ± 0.16%** | 1697.4 | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **AWQ-4** | 91.1% | 89.9% | 92.2% | **91.05% ± 1.14%** | 1685.8 | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **GPTQ-4** | 90.9% | 91.4% | 91.1% | **91.13% ± 0.27%** | 1705.9 | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **BF16** | 88.2% | 88.8% | 89.1% | **88.68% ± 0.46%** | 1769.6 | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **FP8** | 89.1% | 89.2% | 88.1% | **88.80% ± 0.62%** | 1723.7 | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **AWQ-4** | 87.3% | 86.9% | 87.1% | **87.11% ± 0.23%** | 1753.3 | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **GPTQ-4** | 88.5% | 88.6% | 89.8% | **88.96% ± 0.70%** | 1841.5 | 0 | 0 | ✅ COMPLETED |

### C. GPQA-Diamond Breadth Matrix ($n=198$, 3 Seeds: 42, 43, 44)
Per-seed accuracies and mean tokens from `results/README.md` and `results/reports/multitask_benchmark_summary.json`. Do not use earlier scrambled seed rows.
| Model | Quant Format | Seed 42 | Seed 43 | Seed 44 | Mean ± Std | Mean Tokens | Truncations | Repetition Loops | Status |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen-7B** | **BF16** | 51.52% | 46.97% | 52.53% | **50.34% ± 2.96%** | 8529.4 | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **FP8** | 49.49% | 51.01% | 47.98% | **49.49% ± 1.52%** | 8046.6 | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **AWQ-4** | 44.44% | 41.92% | 47.98% | **44.78% ± 3.04%** | 8493.7 | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **GPTQ-4** | 46.97% | 50.00% | 46.97% | **47.98% ± 1.75%** | 9128.3 | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **BF16** | 43.94% | 46.97% | 47.47% | **46.13% ± 1.91%** | 8662.4 | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **FP8** | 47.47% | 47.98% | 47.98% | **47.81% ± 0.29%** | 8718.1 | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **AWQ-4** | 46.97% | 44.95% | 48.99% | **46.97% ± 2.02%** | 8889.2 | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **GPTQ-4** | 44.44% | 40.91% | 49.49% | **44.95% ± 4.32%** | 9247.8 | 0 | 0 | ✅ COMPLETED |

---

## 3. Master TODO List

### [x] Phase 0–4: Completed Foundation & Confirmatory 40-Cell Grid (MATH-500)
* [x] **Phase 0–4:** All 40 MATH-500 cells completed. Pathology after P0 reanalysis: 25 loops / 0 exact cap hits / 209 near-cap on the full 88-cell grid — **not** 0/0.

### [x] Phase 5: Frozen Statistical Analysis & Calibration
* [x] **P5.1 Statistical Hypothesis Testing:** Paired McNemar on maj@5 ($p > 0.29$ after Holm–Bonferroni; not a pass@1-mean test).
* [x] **P5.2 Sample-Consistency Calibration:** ECE, Brier score, AURC computed (gold-hit fraction vs maj@5).
* [x] **P5.3 Systems Telemetry & Cost-of-Pass:** Modeled $C_{\text{pass}}$ frontier at $\$1.50$/h and $65$ tok/s.
* [x] **P5.4 Structured Trace Audit:** 200-sample audit completed.

### [x] Phase 4 Extension: Breadth Benchmark Evaluation
* [x] **P4.Ext.2 GSM8K:** $n=1,319$ grade-school math (4 formats $\times$ 3 seeds) — **100% COMPLETED**.
* [x] **P4.Ext.1 GPQA-Diamond:** $n=198$ expert science (4 formats $\times$ 3 seeds) — **100% COMPLETED**.

### [x] Phase 6: Manuscript Completion & Submission Packaging
* [x] Populate [`paper/main.md`](paper/main.md) with finalized tables, figures, and statistical tests.
* [x] Add comprehensive **Limitations Section**.
* [x] Integrate GSM8K and GPQA-Diamond breadth tables into manuscript.
* [x] Compile publication LaTeX manuscript into [`paper/main.pdf`](paper/main.pdf) (12 pages).
* [x] ArXiv source package [`paper/arxiv_source.zip`](paper/arxiv_source.zip). Journal submission is a later step.

---

## 4. Summary Progress Tracker

| Stage | Milestone | Status | Blockers / Next Steps |
|---|---|---|---|
| **Phase 0–4** | Confirmatory 40-cell grid (MATH-500, seeds 42–46) | ✅ COMPLETED | Validated & backed up |
| **Phase 5** | Frozen statistical analysis, calibration & cost | ✅ COMPLETED | Reports & figures generated |
| **Phase 4 Ext (GSM8K)** | GSM8K breadth evaluation ($n=1,319$, 24 cells) | ✅ COMPLETED | Validated & backed up |
| **Phase 4 Ext (GPQA)** | GPQA-Diamond breadth evaluation ($n=198$, 24 cells) | ✅ COMPLETED | Validated & backed up |
| **Phase 6** | ArXiv manuscript (`paper/main.tex`, `paper/main.pdf`, `paper/arxiv_source.zip`) | ✅ COMPLETED | Pending author review, then MacBook commit/push |

