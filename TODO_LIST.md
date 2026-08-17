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
* **Measured Serving Systems Benchmark:** 🔄 **ACTIVE** on branch `paper-measured-serving` (8 configurations on NVIDIA A100 GPUs).
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

### [ ] 2. Measured Serving Performance Systems Benchmark (IN PROGRESS)
* [x] Created experiment branch `paper-measured-serving`.
* [x] Froze serving protocol in [`docs/MEASURED_SERVING_PROTOCOL.md`](docs/MEASURED_SERVING_PROTOCOL.md).
* [x] Generated stratified 100-prompt benchmark subset in `results/measured_serving/input_subset.json` (seed 20260816, 20 per level).
* [x] Implemented benchmark runner [`scripts/hpc/qrm_parity/benchmark_serving.py`](scripts/hpc/qrm_parity/benchmark_serving.py).
* [x] Implemented submission pipeline [`scripts/hpc/qrm_parity/run_measured_serving.sh`](scripts/hpc/qrm_parity/run_measured_serving.sh).
* [x] Implemented validation audit [`scripts/hpc/qrm_parity/validate_measured_serving.py`](scripts/hpc/qrm_parity/validate_measured_serving.py).
* [x] Implemented analysis engine [`scripts/analysis/measured_serving_analysis.py`](scripts/analysis/measured_serving_analysis.py).
* [x] Launched 8 benchmark jobs on PARAM Rudra HPC across 2 parallel pipelines (1 GPU each, max 2 GPUs).
* [ ] **Next:** Wait for all 8 benchmark jobs to complete.
* [ ] **Next:** Run `validate_measured_serving.py` to audit 48 task-realistic + 8 microbenchmark runs.
* [ ] **Next:** Run `measured_serving_analysis.py` to generate `measured_serving_report.json` and `measured_serving_report.md`.
* [ ] **Next:** Commit benchmark artifacts to `paper-measured-serving` and push to GitHub.

### [ ] 3. Manuscript & Production Finalization (UPCOMING)
* [ ] Rsync serving benchmark results to MacBook.
* [ ] Update Table 6 (Serving Systems & Cost Frontier) in `paper/main.tex` and `paper/main.md` with measured throughput, latency, VRAM, and Cost-of-Pass.
* [ ] Recompile publication PDF [`paper/main.pdf`](paper/main.pdf) with XeLaTeX.
* [ ] Final author review before journal submission to *Future Generation Computer Systems (FGCS)* / *Journal of Systems and Software (JSS)*.
