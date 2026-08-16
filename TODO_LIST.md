# TODO List & Experiment Roadmap — Publication Readiness
**Project:** Reasoning Compression Lab (`reasoning-compression-lab`)  
**Target:** Paper 1 Publication Readiness (Quantization, Reliability, and Cost Frontier for Reasoning Models)  
**Status:** **ALL CAMPAIGNS 100% COMPLETED — 88 Cells (MATH-500, GSM8K, GPQA-Diamond) Backed Up & Manuscript Compiled**  
**Controlling Reference:** [docs/PUBLICATION_READINESS.md](docs/PUBLICATION_READINESS.md) · [docs/plans/2026-08-14-publication-recovery.md](docs/plans/2026-08-14-publication-recovery.md) · [AGENTS.md](AGENTS.md)

---

## 1. Executive Summary & Gating Decision

* **Headline MATH-500 ($n=500$, Seeds 42–46, 40 Cells):** ✅ 100% Completed, validated, and backed up.
* **Breadth Benchmark 1 — GSM8K ($n=1,319$, Seeds 42–44, 24 Cells):** ✅ 100% Completed, validated, and backed up.
* **Breadth Benchmark 2 — GPQA-Diamond ($n=198$, Seeds 42–44, 24 Cells):** ✅ 100% Completed, validated, and backed up.
* **Manuscript Status:** [`paper/main.md`](paper/main.md), [`paper/main.tex`](paper/main.tex), and camera-ready [`paper/main.pdf`](paper/main.pdf) fully drafted and compiled with all 8 tables and formal equations.

---

## 2. Completed Experiments & Verified Campaign Matrices

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
| Model | Quant Format | Seed 42 | Seed 43 | Seed 44 | Mean ± Std | Mean Tokens | Truncations | Repetition Loops | Status |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen-7B** | **BF16** | 53.0% | 51.0% | 47.0% | **50.34% ± 2.96%** | 8153.2 | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **FP8** | 48.0% | 51.0% | 49.5% | **49.49% ± 1.52%** | 7877.9 | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **AWQ-4** | 44.4% | 48.0% | 41.9% | **44.78% ± 3.04%** | 8326.6 | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **GPTQ-4** | 46.5% | 47.5% | 50.0% | **47.98% ± 1.75%** | 8196.4 | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **BF16** | 48.0% | 46.5% | 43.9% | **46.13% ± 1.91%** | 8780.2 | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **FP8** | 48.0% | 47.5% | 48.0% | **47.81% ± 0.29%** | 8632.7 | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **AWQ-4** | 46.5% | 49.2% | 45.2% | **46.97% ± 2.02%** | 8774.2 | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **GPTQ-4** | 40.4% | 44.9% | 49.5% | **44.95% ± 4.32%** | 8916.1 | 0 | 0 | ✅ COMPLETED |

---

## 3. Master TODO List

### [x] Phase 0–4: Completed Foundation & Confirmatory 40-Cell Grid (MATH-500)
* [x] **Phase 0–4:** All 40 cells on MATH-500 completed with 0 truncations and 0 loops.

### [x] Phase 5: Frozen Statistical Analysis & Calibration
* [x] **P5.1 Statistical Hypothesis Testing:** Paired McNemar tests ($p > 0.05$).
* [x] **P5.2 Sample-Consistency Calibration:** ECE, Brier score, AURC computed.
* [x] **P5.3 Systems Telemetry & Cost-of-Pass:** Empirical $C_{\text{pass}}$ frontier established.
* [x] **P5.4 Structured Trace Audit:** 200-sample audit completed.

### [x] Phase 4 Extension: Breadth Benchmark Evaluation
* [x] **P4.Ext.2 GSM8K:** $n=1,319$ grade-school math (4 formats $\times$ 3 seeds) — **100% COMPLETED**.
* [x] **P4.Ext.1 GPQA-Diamond:** $n=198$ expert science (4 formats $\times$ 3 seeds) — **100% COMPLETED**.

### [x] Phase 6: Manuscript Completion & Submission Packaging
* [x] Populate [`paper/main.md`](paper/main.md) with finalized tables, figures, and statistical tests.
* [x] Add comprehensive **Limitations Section**.
* [x] Integrate GSM8K and GPQA-Diamond breadth tables into manuscript.
* [x] Compile publication LaTeX manuscript into [`paper/main.pdf`](paper/main.pdf).
* [x] Journal submission packaging for target Q1 venue (*Future Generation Computer Systems* or *Journal of Systems and Software*).

---

## 4. Summary Progress Tracker

| Stage | Milestone | Status | Blockers / Next Steps |
|---|---|---|---|
| **Phase 0–4** | Confirmatory 40-cell grid (MATH-500, seeds 42–46) | ✅ COMPLETED | Validated & backed up |
| **Phase 5** | Frozen statistical analysis, calibration & cost | ✅ COMPLETED | Reports & figures generated |
| **Phase 4 Ext (GSM8K)** | GSM8K breadth evaluation ($n=1,319$, 24 cells) | ✅ COMPLETED | Validated & backed up |
| **Phase 4 Ext (GPQA)** | GPQA-Diamond breadth evaluation ($n=198$, 24 cells) | ✅ COMPLETED | Validated & backed up |
| **Phase 6** | Final manuscript draft (`paper/main.md`, `paper/main.pdf`) | ✅ COMPLETED | Manuscript compiled & ready for submission |

