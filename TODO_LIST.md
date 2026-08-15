# TODO List & Experiment Roadmap — Publication Readiness
**Project:** Reasoning Compression Lab (`reasoning-compression-lab`)  
**Target:** Paper 1 Publication Readiness (Quantization, Reliability, and Cost Frontier for Reasoning Models)  
**Status:** **Phase 5 Active — Confirmatory 40-Cell Grid Completed**  
**Controlling Reference:** [docs/PUBLICATION_READINESS.md](file:///scratch/manishn_iitp/reasoning-compression-lab/docs/PUBLICATION_READINESS.md) · [docs/plans/2026-08-14-publication-recovery.md](file:///scratch/manishn_iitp/reasoning-compression-lab/docs/plans/2026-08-14-publication-recovery.md) · [AGENTS.md](file:///scratch/manishn_iitp/reasoning-compression-lab/AGENTS.md)

---

## 1. Executive Summary & Gating Decision

* **Current Status (2026-08-15):** The full **40-cell headline confirmatory grid** ($n=500$, seeds 42–46, across BF16, FP8, AWQ-4, GPTQ-4 on Qwen-7B and Llama-8B) is **100% completed and validated** with **0 length truncations** and **0 repetition loops** in `outputs-hpc-campaign-2026-08-14/validation/`.
* **Scientific Finding:** 
  1. **FP8 Parity:** FP8 achieves 100% statistical parity with BF16 across both architectures (Qwen: 94.40% vs 94.00%; Llama: 89.52% vs 89.24%).
  2. **4-Bit Quantization Resilience:** 4-bit GPTQ and AWQ retain exceptional reasoning fidelity on Qwen-7B (>93.1% vs 94.0% BF16), while Llama-8B exhibits greater sensitivity to 4-bit AWQ compression (86.48% vs 89.24% BF16).
  3. **Zero Pathological Degeneration:** Under the pinned `qrm-official` protocol, all 40 cells achieved **0 length truncations** and **0 infinite repetition loops** with >99% answer extraction rate.
* **Active Focus:** Phase 5 Frozen Statistical Analysis (McNemar tests, Holm-Bonferroni correction, sample-consistency calibration on maj@5 subset, cost-per-correct analysis, and trace audit).

---

## 2. Completed Experiments & Verified Campaign Matrix (WHAT IS DONE)

| Model | Quant Format | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 | Mean ± Std | Truncations | Repetition Loops | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen-7B** | **BF16** | 94.4% (472) | 94.0% (470) | 93.8% (469) | 94.6% (473) | 93.2% (466) | **94.00% ± 0.53%** | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **FP8** | 94.4% (472) | 95.2% (476) | 94.8% (474) | 92.6% (463) | 95.0% (475) | **94.40% ± 1.05%** | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **AWQ-4** | 92.4% (462) | 92.8% (464) | 93.2% (466) | 93.0% (465) | 94.2% (471) | **93.12% ± 0.68%** | 0 | 0 | ✅ COMPLETED |
| **Qwen-7B** | **GPTQ-4** | 93.8% (469) | 92.6% (463) | 93.4% (467) | 94.6% (473) | 93.0% (465) | **93.48% ± 0.77%** | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **BF16** | 89.0% (445) | 88.4% (442) | 90.2% (451) | 89.8% (449) | 88.8% (444) | **89.24% ± 0.73%** | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **FP8** | 89.0% (445) | 89.6% (448) | 88.6% (443) | 89.2% (446) | 91.2% (456) | **89.52% ± 1.02%** | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **AWQ-4** | 84.4% (422) | 84.8% (424) | 89.2% (446) | 87.4% (437) | 86.6% (433) | **86.48% ± 1.95%** | 0 | 0 | ✅ COMPLETED |
| **Llama-8B** | **GPTQ-4** | 88.0% (440) | 89.6% (448) | 86.8% (434) | 89.4% (447) | 90.8% (454) | **88.92% ± 1.54%** | 0 | 0 | ✅ COMPLETED |

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

## 4. Master TODO List: What is Yet to Be Done

### [x] Phase 0–4: Completed Foundation & Confirmatory 40-Cell Grid
* [x] **Phase 0:** Clean patch recreation, schema extension, and 3-question smoke tests.
* [x] **Phase 1:** Matched BF16 vs FP8 seed-42 baseline.
* [x] **Phase 2:** Discriminating 3-seed pilot across 4 formats (Seeds 42, 43, 44).
* [x] **Phase 3:** Contribution selection gate passed (Track A: Quantization Reliability & Cost Frontier).
* [x] **Phase 4:** Headline 5-seed confirmatory grid (Seeds 42–46) across all 8 model-format configurations.

---

### [ ] Phase 5: Frozen Statistical Analysis & Calibration (CURRENT PRIORITY)
* [ ] **P5.1 Statistical Hypothesis Testing:**
  * [ ] Compute paired McNemar tests and 95% Wilson / bootstrap confidence intervals comparing BF16 vs FP8, AWQ-4, and GPTQ-4.
  * [ ] Apply Holm-Bonferroni correction across tests.
  * [ ] Separate seed-to-seed variance from problem sampling error.
* [ ] **P5.2 Sample-Consistency Calibration:**
  * [ ] Execute `maj@5` on the predeclared 100-problem stratified subset.
  * [ ] Calculate Brier score, Expected Calibration Error (ECE), and Area Under Risk-Coverage curve (AURC).
* [ ] **P5.3 Systems Telemetry & Cost-of-Pass:**
  * [ ] Compute generation throughput (tokens/sec), latency distributions ($p_{50}, p_{95}, p_{99}$), and peak VRAM.
  * [ ] Calculate Cost-of-Pass ($C_{\text{pass}}$ / cost-per-correct answer) under explicit GPU cloud pricing models ($/GPU-hr).
* [ ] **P5.4 Structured Trace Audit:**
  * [ ] Perform manual review of $\ge 200$ stratified completions.
  * [ ] Verify reasoning trace integrity and qualitative step correctness.
* **Gate P5 Criteria:** All tables, figures, and statistical tests generated reproducibly via Python scripts.

---

### [ ] Phase 4 Extension: Breadth Benchmark Evaluation (Gated)
* [ ] **P4.Ext.1 GPQA-Diamond:**
  * [ ] $n=198$ expert science reasoning items (zero-shot, 4 formats $\times$ 3 seeds: 42, 43, 44).
* [ ] **P4.Ext.2 GSM8K:**
  * [ ] $n=1,319$ grade-school math reasoning items (zero-shot, 4 formats $\times$ 3 seeds: 42, 43, 44).

---

### [ ] Phase 6: Manuscript Completion & Submission Packaging
* [ ] Populate [`paper/main.md`](file:///scratch/manishn_iitp/reasoning-compression-lab/paper/main.md) with finalized tables, figures, and statistical tests.
* [ ] Add comprehensive **Limitations Section** (covering A100 W8A16 Marlin fallback, single-family model scope, and shared-cluster limits).
* [ ] Prepare open-source release artifact bundle:
  * Pinned requirements & conda lockfiles.
  * Patch series in `patches/`.
  * Raw execution manifests and configuration hashes.
  * Evaluation & scoring scripts.
* **Gate P6 Criteria:** Full supervisor sign-off; manuscript is submission-ready for *FGCS* or *JSS*.

---

## 5. Summary Progress Tracker

| Stage | Milestone | Status | Blockers / Next Steps |
|---|---|---|---|
| **Phase 0–4** | Confirmatory 40-cell grid (MATH-500, seeds 42–46) | ✅ COMPLETED | Validated in `outputs-hpc-campaign-2026-08-14/` |
| **Phase 5** | Frozen statistical analysis, calibration & cost | 🔄 ACTIVE | Run paired McNemar, bootstrap CIs, maj@5 calibration |
| **Phase 4 Ext** | Breadth evaluation (GPQA-Diamond, GSM8K) | ⏸️ Ready to Launch | Launch via queue daemon |
| **Phase 6** | Final manuscript draft (`paper/main.md`) & artifact | ⏳ Next | Blocked by Phase 5 analysis |
