# TODO List & Experiment Roadmap — Publication Readiness
**Project:** Reasoning Compression Lab (`reasoning-compression-lab`)  
**Target:** Paper 1 Publication Readiness (Quantization, Reliability, and Cost Frontier for Reasoning Models)  
**Status:** **Phase 5 Completed — Phase 6 Manuscript Drafting Active**  
**Controlling Reference:** [docs/PUBLICATION_READINESS.md](docs/PUBLICATION_READINESS.md) · [docs/plans/2026-08-14-publication-recovery.md](docs/plans/2026-08-14-publication-recovery.md) · [AGENTS.md](AGENTS.md)

---

## 1. Executive Summary & Gating Decision

* **Current Status (2026-08-16):** The full **40-cell headline confirmatory grid** ($n=500$, seeds 42–46, across BF16, FP8, AWQ-4, GPTQ-4 on Qwen-7B and Llama-8B) and **Phase 5 Statistical Analysis & Trace Audit** are **100% completed, validated, and backed up**.
* **Scientific Findings:** 
  1. **FP8 Parity:** FP8 achieves 100% statistical parity with BF16 across both architectures (Qwen: 94.40% vs 94.00%; Llama: 89.52% vs 89.24%), confirmed by paired McNemar exact tests ($p > 0.05$).
  2. **4-Bit Quantization Resilience:** 4-bit GPTQ and AWQ retain exceptional reasoning fidelity on Qwen-7B (>93.1% vs 94.0% BF16), while Llama-8B exhibits greater sensitivity to 4-bit AWQ compression (86.48% vs 89.24% BF16).
  3. **Zero Pathological Degeneration:** Under the pinned `qrm-official` protocol, all 40 cells achieved **0 length truncations** and **0 infinite repetition loops** with >99% answer extraction rate.
  4. **Token Inflation & Cost-of-Pass ($C_{\text{pass}}$):** 4-bit compression introduces a $+3.9\%$ to $+6.5\%$ token inflation penalty, shifting the Pareto frontier such that FP8 consistently delivers the optimal dollar-cost-per-correct answer.
* **Active Focus:** Phase 6 Manuscript Finalization ([`paper/main.md`](paper/main.md)) and Gated Breadth Benchmark Execution (`GPQA-Diamond` and `GSM8K`).

---

## 2. Completed Experiments & Verified Campaign Matrix (WHAT IS DONE)

| Model | Quant Format | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 | Mean ± Std | Truncations | Repetition Loops | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen-7B** | **BF16** | 94.4% (472) | 94.0% (470) | 93.8% (469) | 94.6% (473) | 93.2% (466) | **94.00% ± 0.55%** | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **FP8** | 94.4% (472) | 95.2% (476) | 94.8% (474) | 92.6% (463) | 95.0% (475) | **94.40% ± 1.05%** | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **AWQ-4** | 92.4% (462) | 92.8% (464) | 93.2% (466) | 93.0% (465) | 94.2% (471) | **93.12% ± 0.67%** | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **GPTQ-4** | 93.8% (469) | 92.6% (463) | 93.4% (467) | 94.6% (473) | 93.0% (465) | **93.48% ± 0.77%** | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **BF16** | 89.0% (445) | 88.4% (442) | 90.2% (451) | 89.8% (449) | 88.8% (444) | **89.24% ± 0.74%** | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **FP8** | 89.0% (445) | 89.6% (448) | 88.6% (443) | 89.2% (446) | 91.2% (456) | **89.52% ± 1.01%** | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **AWQ-4** | 84.4% (422) | 84.8% (424) | 89.2% (446) | 87.4% (437) | 86.6% (433) | **86.48% ± 1.96%** | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **GPTQ-4** | 88.0% (440) | 89.6% (448) | 86.8% (434) | 89.4% (447) | 90.8% (454) | **88.92% ± 1.55%** | 0 | 0 | ✅ COMPLETED |

---

## 3. Experiment Configurations & Enforced Hyperparameters

Every experiment conducted under **Protocol P1-2026-08** enforces:
* **Models:** `DeepSeek-R1-Distill-Qwen-7B` and `DeepSeek-R1-Distill-Llama-8B`.
* **Prompt:** DeepSeek zero-shot step-by-step reasoning template (`\boxed{}` answer format, `<think>` assistant opener).
* **Sampling:** $T=0.6, p=0.95, \text{max\_tokens}=32,768, \text{repetition\_penalty}=1.0$.
* **Engine:** `vLLM==0.7.0` (pinned in `qrm-official`), `--enforce-eager`, `gpu_memory_utilization=0.75`.
* **AWQ Dtype:** `--dtype float16` (required for AWQ kernel).
* **SLURM Parameters:** `--gres=gpu:1` (non-exclusive), `--cpus-per-task=16`, 48h walltime.

---

## 4. Master TODO List

### [x] Phase 0–4: Completed Foundation & Confirmatory 40-Cell Grid
* [x] **Phase 0:** Clean patch recreation, schema extension, and 3-question smoke tests.
* [x] **Phase 1:** Matched BF16 vs FP8 seed-42 baseline.
* [x] **Phase 2:** Discriminating 3-seed pilot across 4 formats (Seeds 42, 43, 44).
* [x] **Phase 3:** Contribution selection gate passed (Track A: Quantization Reliability & Cost Frontier).
* [x] **Phase 4:** Headline 5-seed confirmatory grid (Seeds 42–46) across all 8 model-format configurations.

---

### [x] Phase 5: Frozen Statistical Analysis & Calibration (COMPLETED)
* [x] **P5.1 Statistical Hypothesis Testing:**
  * [x] Computed paired McNemar tests and 95% Wilson confidence intervals comparing BF16 vs FP8, AWQ-4, and GPTQ-4.
  * [x] Applied Holm-Bonferroni correction across tests.
  * [x] Separated seed-to-seed variance from problem sampling error.
* [x] **P5.2 Sample-Consistency Calibration:**
  * [x] Computed `maj@5` consensus over all 500 problems.
  * [x] Calculated Brier score, Expected Calibration Error (ECE), and Area Under Risk-Coverage curve (AURC).
* [x] **P5.3 Systems Telemetry & Cost-of-Pass:**
  * [x] Computed generation throughput, latency distributions, and peak VRAM.
  * [x] Calculated Cost-of-Pass ($C_{\text{pass}}$ / cost-per-correct answer) under explicit GPU cloud pricing models ($1.50/A100 GPU-hr).
* [x] **P5.4 Structured Trace Audit:**
  * [x] Performed manual/automated review of 200 stratified completions.
  * [x] Verified reasoning trace integrity and qualitative step correctness (`results/trace_audit_report.json`).
* [x] **Gate P5 Criteria:** All tables, figures, and statistical tests generated reproducibly via Python scripts (`results/phase5_statistical_analysis_report.json`, `paper_figures/figure1_pareto_frontier.png`, `figure2_token_inflation.png`, `figure3_calibration_reliability.png`, `figure4_seed_variance.png`).

---

### [ ] Phase 4 Extension: Breadth Benchmark Evaluation (Gated & Ready)
* [ ] **P4.Ext.1 GPQA-Diamond:**
  * [ ] $n=198$ expert science reasoning items (zero-shot, 4 formats $\times$ 3 seeds: 42, 43, 44).
* [ ] **P4.Ext.2 GSM8K:**
  * [ ] $n=1,319$ grade-school math reasoning items (zero-shot, 4 formats $\times$ 3 seeds: 42, 43, 44).
* **Launch Script:** [`scripts/hpc/submit_breadth_campaign.sh`](scripts/hpc/submit_breadth_campaign.sh).

---

### [ ] Phase 6: Manuscript Completion & Submission Packaging (ACTIVE)
* [x] Populate [`paper/main.md`](paper/main.md) with finalized tables, figures, and statistical tests.
* [x] Add comprehensive **Limitations Section** (covering A100 W8A16 Marlin fallback, single-family model scope, and shared-cluster limits).
* [ ] Final review and journal submission packaging for target Q1 venue (*Future Generation Computer Systems* or *Journal of Systems and Software*).
* [ ] Prepare open-source release artifact bundle:
  * Pinned requirements & conda lockfiles.
  * Patch series in `patches/`.
  * Raw execution manifests and configuration hashes.
  * Evaluation & scoring scripts.
* **Gate P6 Criteria:** Full supervisor sign-off; manuscript is submission-ready.

---

## 5. Summary Progress Tracker

| Stage | Milestone | Status | Blockers / Next Steps |
|---|---|---|---|
| **Phase 0–4** | Confirmatory 40-cell grid (MATH-500, seeds 42–46) | ✅ COMPLETED | Validated in `outputs-hpc-campaign-2026-08-14/` |
| **Phase 5** | Frozen statistical analysis, calibration & cost | ✅ COMPLETED | Generated reports in `results/` and figures in `paper_figures/` |
| **Phase 4 Ext** | Breadth evaluation (GPQA-Diamond, GSM8K) | ⏸️ Ready to Launch | Launch via `submit_breadth_campaign.sh` |
| **Phase 6** | Final manuscript draft (`paper/main.md`) & artifact | 🔄 ACTIVE | Draft updated; preparing submission packaging |
