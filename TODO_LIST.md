# TODO List & Experiment Roadmap — Publication Readiness
**Project:** Reasoning Compression Lab (`reasoning-compression-lab`)  
**Target:** Paper 1 Publication Readiness (Quantization, Reliability, and Cost Frontier for Reasoning Models)  
**Status:** **88-cell campaign complete. Modal agreement complete on `paper-modal-agreement`. Measured serving benchmark active on `paper-measured-serving`.**  
**Controlling Reference:** [docs/PUBLICATION_READINESS.md](docs/PUBLICATION_READINESS.md) · [docs/MEASURED_SERVING_PROTOCOL.md](docs/MEASURED_SERVING_PROTOCOL.md) · [AGENTS.md](AGENTS.md)

---

## 1. Executive Summary & Gating Decision

* **Headline MATH-500 ($n=500$, Seeds 42–46, 40 Cells):** ✅ 100% Completed, validated, and backed up.
* **Breadth Benchmark 1 — GSM8K ($n=1,319$, Seeds 42–44, 24 Cells):** ✅ 100% Completed, validated, and backed up.
* **Breadth Benchmark 2 — GPQA-Diamond ($n=198$, Seeds 42–44, 24 Cells):** ✅ 100% Completed, validated, and backed up.
* **Gold-Free Modal Agreement Analysis:** ✅ 100% Completed, validated, and pushed on branch `paper-modal-agreement` (HEAD `845d879`).
* **Measured Serving Systems Benchmark:** ✅ **Complete** on branch `paper-measured-serving` (48 task-realistic + 8 microbenchmark JSON files; manuscript integrated).
* **Manuscript Status:** Canonical source [`paper/main.tex`](paper/main.tex) compiled to [`paper/main.pdf`](paper/main.pdf).

---

## 2. Completed Experiments & Verified Campaign Matrices

Pathology on the **full 88-cell grid**: **25 loops**, **0 exact cap hits**, **209 near-cap**. See `results/README.md`.

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
| **Qwen-7B** | **BF16** | 51.52% | 46.97% | 52.53% | **50.34% ± 2.96%** | 8529.4 | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **FP8** | 49.49% | 51.01% | 47.98% | **49.49% ± 1.52%** | 8046.6 | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **AWQ-4** | 44.44% | 41.92% | 47.98% | **44.78% ± 3.04%** | 8493.7 | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **GPTQ-4** | 46.97% | 50.00% | 46.97% | **47.98% ± 1.75%** | 9128.3 | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **BF16** | 43.94% | 46.97% | 47.47% | **46.13% ± 1.91%** | 8662.4 | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **FP8** | 47.47% | 47.98% | 47.98% | **47.81% ± 0.29%** | 8718.1 | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **AWQ-4** | 46.97% | 44.95% | 48.99% | **46.97% ± 2.02%** | 8889.2 | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **GPTQ-4** | 44.44% | 40.91% | 49.49% | **44.95% ± 4.32%** | 9247.8 | 0 | 0 | ✅ COMPLETED |

---

## 3. Master TODO List & Immediate Next Steps

### [x] 1. Gold-Free Modal Agreement Analysis (COMPLETED)
* [x] Checked out `paper-modal-agreement` from commit `3076573`.
* [x] Built canonical `scripts/analysis/modal_agreement_analysis.py` with 100% campaign score reproduction (20,000 / 20,000).
* [x] Solved transitivity check with primary parsed prediction (0 symmetry violations, 0 transitivity violations across 4,000 groups).
* [x] Computed risk-coverage frontier across $\ge 3/5$, $\ge 4/5$, and $5/5$ thresholds with 10,000 bootstrap replicates and paired CIs vs BF16.
* [x] Generated `results/recovered/math500_modal_inputs.jsonl` (3.8 MB, SHA256 `23e9ead021111959cf047323572889c95be0496e9475d6870b06c8b2c9a6149b`).
* [x] Committed and pushed branch `paper-modal-agreement` to GitHub (HEAD `845d879`).

### [x] 2. Measured Serving Systems Initial Benchmark (COMPLETED & AUDITED)
* [x] Created experiment branch `paper-measured-serving`.
* [x] Completed initial 56 benchmark runs across 8 configurations on A100.
* [x] Audited benchmark protocol; identified Condition A level balance, Condition B `max_num_seqs=8` pinning, sidecar provenance alignment, and single-node physical control for confirmation.

### [x] 3. Measured Serving Systems Confirmation Benchmark (COMPLETED & 100% VALIDATED)
* [x] Created and verified dedicated branch [`paper-serving-confirmation`](file:///scratch/manishn_iitp/reasoning-compression-lab).
* [x] Authored pre-execution frozen protocol [`docs/MEASURED_SERVING_CONFIRMATION_PROTOCOL.md`](docs/MEASURED_SERVING_CONFIRMATION_PROTOCOL.md) (SHA256 `5f665911...`).
* [x] Generated balanced Condition A subset (20 items, exactly 4 per level 1–5, seed 20260817).
* [x] Generated balanced Condition B subset (100 items, exactly 20 per level 1–5, seed 20260817).
* [x] Verified 100% bijective problem/full_prompt provenance alignment on all 120 subset records.
* [x] Implemented `benchmark_serving_confirmation.py` with explicit `max_num_seqs=8` runtime assertion and $\text{CV} \le 3.0\%$ expansion logic.
* [x] Launched dual-GPU parallel execution on PARAM Rudra HPC (Job `96766` on `ragpu003` for Qwen-7B; Job `96768` on `ragpu004` for Llama-8B).
* [x] Verified zero intra-architecture node mixing (100% single-node physical control per model family: Qwen on `ragpu003`, Llama on `ragpu004`).
* [x] Activated detached 30-minute Telegram heartbeat & milestone daemon (`telegram_30min_heartbeat.py`).
* [x] Completed all 60 confirmation runs across all 8 configurations with 0 OOMs, 0 crashes, and $\text{CV} \le 3.0\%$.

### [/] 4. Post-Confirmation Sequence & Paper 1 (J1) Submission Protocol
* [x] **Step 1: Automated Audit & Cost Remodeling (HPC):**
  * Executed `validate_measured_serving_confirmation.py` (asserted 60 runs, 0 OOMs, single-node physical invariance, 0 integrity errors).
  * Executed `measured_serving_confirmation_analysis.py` (computed measured Cost-of-Pass $C_{\text{pass}}^{\text{meas}}$ and generated JSON + Markdown reports).
  * Sent completion alerts to Telegram bot (`@ManMan06`).
* [ ] **Step 2: Commit & GitHub Push (HPC):**
  * Stage all confirmation artifacts, reports, and code to branch `paper-serving-confirmation` and push to GitHub.
* [ ] **Step 3: MacBook Manuscript Sync & PDF Compilation:**
  * Pull confirmation results to MacBook via `scripts/macbook/rsync_from_hpc.sh`.
  * Update Tables in `paper/main.tex` and `paper/main.md` with verified confirmation values.
  * Recompile camera-ready `paper/main.pdf` (7 pages).
* [ ] **Step 3: `finish_reason` & Truncation Metrology Diagnostic (HPC):**
  * Evaluate generation termination dynamics (`stop` token vs `length` truncation) across 4k, 8k, 16k, 32k token budgets.
* [ ] **Step 4: Official Journal Submission (Paper 1):**
  * Target venue: *Future Generation Computer Systems (FGCS)* / *Journal of Systems and Software (JSS)* (Scopus/SJR Q1).
  * Package reproduction bundle (`paper/arxiv_source.zip` + Zenodo schema).
  * Prepare 2-page monthly supervisor briefing document.
* [ ] **Step 5: PhD Thesis Pivot — Paper 2 (J2): Reasoning Speculative Decoding:**
  * Train and benchmark small draft models (0.5B–1.5B) for speculative reasoning acceleration on HPC 2× A100.

