# TODO List & Experiment Roadmap — Publication Readiness
**Project:** Reasoning Compression Lab (`reasoning-compression-lab`)  
**Target:** Paper 1 Publication Readiness (Quantization, Reliability, and Cost Frontier for Reasoning Models)  
**Status:** **Phase 5 & GSM8K Completed — GPQA-Diamond Breadth Campaign Active (24/7 Daemon)**  
**Controlling Reference:** [docs/PUBLICATION_READINESS.md](docs/PUBLICATION_READINESS.md) · [docs/plans/2026-08-14-publication-recovery.md](docs/plans/2026-08-14-publication-recovery.md) · [AGENTS.md](AGENTS.md)

---

## 1. Executive Summary & Gating Decision

* **Headline MATH-500 ($n=500$, Seeds 42–46, 40 Cells):** 100% Completed, validated, and backed up.
* **Breadth Benchmark 1 — GSM8K ($n=1,319$, Seeds 42–44, 24 Cells):** 100% Completed, validated, and backed up.
  - **Qwen-7B:** BF16: 91.26% ± 0.23% | FP8: 91.33% ± 0.13% | AWQ-4: 91.05% ± 0.93% | GPTQ-4: 91.13% ± 0.22%
  - **Llama-8B:** BF16: 88.68% ± 0.38% | FP8: 88.80% ± 0.50% | AWQ-4: 87.11% ± 0.19% | GPTQ-4: 88.96% ± 0.58%
* **Breadth Benchmark 2 — GPQA-Diamond ($n=198$, Seeds 42–44, 24 Cells):** 🔄 **RUNNING** on HPC via autonomous queue daemon `gpqa_daemon`.
* **Manuscript Status:** [`paper/main.md`](paper/main.md) updated with full literature references, trace audit analysis, and mathematical formulations.

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
| **Qwen-7B** | **BF16** | 91.1% | 91.6% | 91.1% | **91.26% ± 0.23%** | 1695.9 | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **FP8** | 91.3% | 91.5% | 91.2% | **91.33% ± 0.13%** | 1697.4 | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **AWQ-4** | 91.1% | 89.9% | 92.2% | **91.05% ± 0.93%** | 1685.8 | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **GPTQ-4** | 90.9% | 91.4% | 91.1% | **91.13% ± 0.22%** | 1705.9 | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **BF16** | 88.2% | 88.8% | 89.1% | **88.68% ± 0.38%** | 1769.6 | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **FP8** | 89.1% | 89.2% | 88.1% | **88.80% ± 0.50%** | 1723.7 | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **AWQ-4** | 87.3% | 86.9% | 87.1% | **87.11% ± 0.19%** | 1753.3 | 0 | 1 | ✅ COMPLETED |
| **Llama-8B** | **GPTQ-4** | 88.5% | 88.6% | 89.8% | **88.96% ± 0.58%** | 1841.5 | 0 | 0 | ✅ COMPLETED |

---

## 3. Master TODO List

### [x] Phase 0–4: Completed Foundation & Confirmatory 40-Cell Grid (MATH-500)
* [x] **Phase 0–4:** All 40 cells on MATH-500 completed with 0 truncations and 0 loops.

### [x] Phase 5: Frozen Statistical Analysis & Calibration
* [x] **P5.1 Statistical Hypothesis Testing:** Paired McNemar tests ($p > 0.05$).
* [x] **P5.2 Sample-Consistency Calibration:** ECE, Brier score, AURC computed.
* [x] **P5.3 Systems Telemetry & Cost-of-Pass:** Empirical $C_{\text{pass}}$ frontier established.
* [x] **P5.4 Structured Trace Audit:** 200-sample audit completed.

### [ ] Phase 4 Extension: Breadth Benchmark Evaluation
* [x] **P4.Ext.2 GSM8K:** $n=1,319$ grade-school math (4 formats $\times$ 3 seeds) — **100% COMPLETED**.
* [ ] **P4.Ext.1 GPQA-Diamond:** $n=198$ expert science (4 formats $\times$ 3 seeds) — **RUNNING in `gpqa_daemon`**.

### [ ] Phase 6: Manuscript Completion & Submission Packaging
* [x] Populate [`paper/main.md`](paper/main.md) with finalized tables, figures, and statistical tests.
* [x] Add comprehensive **Limitations Section**.
* [ ] Integrate GSM8K and GPQA-Diamond breadth tables into manuscript once GPQA finishes.
* [ ] Journal submission packaging for target Q1 venue (*Future Generation Computer Systems* or *Journal of Systems and Software*).

---

## 4. Summary Progress Tracker

| Stage | Milestone | Status | Blockers / Next Steps |
|---|---|---|---|
| **Phase 0–4** | Confirmatory 40-cell grid (MATH-500, seeds 42–46) | ✅ COMPLETED | Validated & backed up |
| **Phase 5** | Frozen statistical analysis, calibration & cost | ✅ COMPLETED | Reports & figures generated |
| **Phase 4 Ext (GSM8K)** | GSM8K breadth evaluation ($n=1,319$, 24 cells) | ✅ COMPLETED | Validated & backed up |
| **Phase 4 Ext (GPQA)** | GPQA-Diamond breadth evaluation ($n=198$, 24 cells) | 🔄 RUNNING | Autonomous daemon active in tmux |
| **Phase 6** | Final manuscript draft (`paper/main.md`) & artifact | 🔄 ACTIVE | Draft updated; awaiting GPQA completion |
