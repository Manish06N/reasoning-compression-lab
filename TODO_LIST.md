# TODO List & Experiment Roadmap — Publication Readiness

**Project:** Reasoning Compression Lab (`reasoning-compression-lab`)  
**Target:** Paper 1 Publication Readiness (Quantization, Reliability, and Cost Frontier for Reasoning Models)  
**Status:** **Needs Revision — Recovery Phase Active**  
**Controlling Reference:** [docs/PUBLICATION_READINESS.md](file:///scratch/manishn_iitp/reasoning-compression-lab/docs/PUBLICATION_READINESS.md) · [docs/plans/2026-08-14-publication-recovery.md](file:///scratch/manishn_iitp/reasoning-compression-lab/docs/plans/2026-08-14-publication-recovery.md)

---

## 1. Executive Summary & Gating Decision

* **Current Status:** Jobs **96100** (Qwen-7B FP8) and **96101** (Llama-8B FP8) successfully completed $n=500$ MATH-500 runs.
* **Scientific Verdict:** Validated replication/control evidence reproducing public model cards (Qwen 94.4%, Llama 89.0%). **Not yet publication-ready** as a causal quantization finding because:
  1. No matched BF16 run on the identical pinned stack.
  2. Evaluated on a single seed (42) and single task (MATH-500).
  3. Output schema lacks `finish_reason` and token IDs (trace audit revealed 6 near-cap completions and phrase loops).
  4. Hardware context: NVIDIA A100 executes FP8 checkpoints via vLLM's **Marlin weight-only fallback (W8A16)**, not native FP8 compute.
  5. Upstream `external/Quantized-Reasoning-Models` dependency relies on uncommitted local patches.
* **Mandate:** All broad grid runs (`b03`, `b04`, etc.) remain **BLOCKED** until Phase 0 (reproducibility & observability) is completed and verified.

---

## 2. Completed Experiments & Current Evidence (WHAT IS DONE)

| Job ID | Model | Format | Dataset | Seed | Stack | Key Result | Role in Paper | Status |
|---|---|---|---|---|---|---|---|---|
| **87302** | Qwen-7B | BF16 | MATH-500 ($n=10$) | 42 | `qrm-official` (vLLM 0.7.0) | **10/10 (100%)**, 0% trunc | Protocol validity anchor | ✅ COMPLETED |
| **86758** | Llama-8B | BF16 | MATH-500 ($n=500$) | 0 | `qreason` (vLLM 0.8.5) | **19.6%**, 58% trunc (`sober` prompt) | Modern stack diagnostic | ⚠️ ARCHIVED |
| **86757** | Qwen-7B | BF16 | MATH-500 ($n=410$) | 0 | `qreason` (vLLM 0.8.5) | **~7%**, 94.1% trunc | Modern stack diagnostic | ⚠️ ARCHIVED |
| **87116/17** | Qwen/Llama | BF16 | MATH-500 ($n=20$) | 42 | `qreason` (Path C) | 10–15% pass@1, 75–90% trunc | Stack gap proof | ⚠️ ARCHIVED |
| **96093** | Qwen-7B | FP8 | MATH-500 ($n=10$) | 42 | `qrm-official` (vLLM 0.7.0) | **10/10 (100%)**, 0% trunc, 0 loops | Exact-stack pilot gate | ✅ COMPLETED |
| **96094** | Llama-8B | FP8 | MATH-500 ($n=10$) | 42 | `qrm-official` (vLLM 0.7.0) | **10/10 (100%)**, 0% trunc, 0 loops | Exact-stack pilot gate | ✅ COMPLETED |
| **96100** | Qwen-7B | FP8 | MATH-500 ($n=500$) | 42 | `qrm-official` (vLLM 0.7.0) | **472/500 (94.4%)** [92.03–96.10%] | Appendix / Control Table | ✅ COMPLETED |
| **96101** | Llama-8B | FP8 | MATH-500 ($n=500$) | 42 | `qrm-official` (vLLM 0.7.0) | **445/500 (89.0%)** [85.95–91.45%] | Appendix / Control Table | ✅ COMPLETED |
| **96237** | Qwen-7B | BF16 | MATH-500 ($n=500$) | 42 | `qrm-official` (vLLM 0.7.0) | **472/500 (94.4%)** | Phase 1 Headline Matched Control | ✅ COMPLETED |
| **96238** | Qwen-7B | FP8 | MATH-500 ($n=500$) | 42 | `qrm-official` (vLLM 0.7.0) | **472/500 (94.4%)** | Phase 1 Headline Matched Control | ✅ COMPLETED |
| **96240** | Qwen-7B | GPTQ-4 | MATH-500 ($n=500$) | 42 | `qrm-official` (vLLM 0.7.0) | **469/500 (93.8%)** | Phase 2 4-bit Quantization Pilot | ✅ COMPLETED |
| **96289** | Qwen-7B | AWQ-4 | MATH-500 ($n=500$) | 42 | `qrm-official` (vLLM 0.7.0) | *In Progress (float16)* | Phase 2 4-bit Quantization Pilot | 🔄 RUNNING |
| **96247** | Llama-8B | FP8 | MATH-500 ($n=500$) | 42 | `qrm-official` (vLLM 0.7.0) | *In Progress* | Phase 1 Llama FP8 Control | 🔄 RUNNING |

---

## 3. Experiment Configurations & Fixed Hyperparameters

Every experiment conducted under the publication protocol (**Protocol P1-2026-08**) must enforce the following exact parameters:

### A. Model & Checkpoint Configurations
* **Qwen Headline Anchor:** `DeepSeek-R1-Distill-Qwen-7B`
  * Checkpoint revisions: BF16 base, FP8 (`ceb2fcd1...`), AWQ4, GPTQ4.
* **Llama Headline Anchor:** `DeepSeek-R1-Distill-Llama-8B`
  * Checkpoint revisions: BF16 base, FP8 (`5d548d91...`), AWQ4, GPTQ4.

### B. Prompt & Decoding Parameters (Protocol P1-2026-08)
* **Prompt Template:** DeepSeek reasoning template (zero-shot, no system prompt, math problem with step-by-step instruction, `\boxed{}` answer format, assistant opens with `<think>`).
* **Sampling Temperature:** `0.6`
* **Top-p:** `0.95`
* **Max Generation Tokens (`max_new_tokens`):** `32768`
* **Max Model Length (`max_model_len`):** `32768`
* **Repetition Penalty:** `1.0` (None)
* **Sampling Seeds:**
  * Pilot Seeds: `42`, `43`, `44`
  * Headline Confirmatory Seeds: `42`, `43`, `44`, `45`, `46`

### C. Serving & Engine Settings
* **Inference Engine:** `vLLM==0.7.0` (pinned in `qrm-official`)
* **Execution Mode:** `--enforce-eager` (Required: avoids runtime Triton/Inductor JIT link failures on HPC compute nodes)
* **Parallelism:** Tensor Parallel = `1`, Pipeline Parallel = `1`
* **KV Cache Dtype:** `auto`
* **Prefix Caching:** `False`
* **Chunked Prefill:** `False`
* **GPU Memory Utilization:** `0.75` (safely reserves 60 GB on 80 GB A100 to prevent shared-GPU OOM)

### D. Cluster & SLURM Resource Allocation
* **Partition:** `gpu`
* **Resource Request:** `--gres=gpu:1` (Strict rule: **NEVER** use `--exclusive` for 1-GPU cells; `--exclusive` consumes the full 2-GPU user quota)
* **CPUs per task:** `--cpus-per-task=16` (Ensures proportional host RAM allocation)
* **Node pool:** `ragpu[003-008]`, `racn[115-116]` (A100 80GB PCIe)
* **Walltime limit:** 48 hours

---

## 4. Master TODO List: What is Yet to Be Done

### [ ] Phase 0: Reproducibility & Observability Infrastructure (NO GPU JOBS EXCEPT 3-Q SMOKE)
* [ ] **P0.1 Tracked Upstream Patches:**
  * [ ] Extract all uncommitted local modifications from `external/Quantized-Reasoning-Models/` into explicit `.patch` files inside [`patches/`](file:///scratch/manishn_iitp/reasoning-compression-lab/patches/).
  * [ ] Create [`configs/external_repo_pins.json`](file:///scratch/manishn_iitp/reasoning-compression-lab/configs/external_repo_pins.json) with exact commit SHAs and patch series.
  * [ ] Update `setup_official_qrm_repo.sh` to deterministically apply patches on clean checkouts.
* [ ] **P0.2 Schema & Provenance Extension:**
  * [ ] Extend output JSONL schema to capture:
    * `finish_reason` (`stop`, `length`, `abort`)
    * `completion_tokens`, `prompt_tokens`, `total_tokens`
    * `output_token_ids` (or hash/sample for loop verification)
    * `request_latency_ms`, `time_to_first_token_ms`
    * `model_commit_hash`, `prompt_hash`, `dataset_revision`
    * `peak_vram_mb`, `gpu_energy_joules` (where supported)
* [ ] **P0.3 Pathology & Multi-Level Validation:**
  * [ ] Implement 3 distinct validation tiers in scoring scripts:
    1. `integrity_pass` (structural integrity, hash verification, 500/500 rows, prompt/gold match)
    2. `quality_warnings` (near-cap length endings, n-gram repetition, phrase-level loops, parse failures)
    3. `publication_gate` (matched comparison present, required seed count, confidence verified)
  * [ ] Add phrase/sentence-level loop detection to catch degenerations missed by simple 1-word detectors.
* [ ] **P0.4 Controlled Telemetry Instrumentation:**
  * [ ] Add background `nvidia-smi` VRAM & power polling per request with graceful fallback (explicit `null` / error state instead of zero-filling).
* [ ] **P0.5 Automated Tests & Smoke Validation:**
  * [ ] Write unit tests for schema validation, pathology detection, and patch application.
  * [ ] Run CPU-only test suite (`pytest tests/`).
  * [ ] Run 3-question GPU smoke test across each format (BF16, FP8, AWQ4, GPTQ4) and verify all output fields are populated.
* **Gate P0 Criteria:** Clean clone reproduction passes in a temporary directory; tests pass; 3-question smoke produces full schema with valid `finish_reason`.

---

### [ ] Phase 1: Matched BF16 vs FP8 Pilot (4 Cells on MATH-500, Seed 42)
* **Goal:** Scientifically isolate the direct difference between BF16 and FP8 when evaluated under identical prompt, stack, decoding, and scoring configurations.
* **Task:** `HuggingFaceH4/MATH-500` ($n=500$)
* **Sampling Seed:** `42`
* **Cell Matrix (4 Jobs):**
  1. [ ] `pilot_qwen7b_bf16_math500_seed42` (Qwen-7B BF16)
  2. [ ] `pilot_qwen7b_fp8_math500_seed42` (Qwen-7B FP8 — rerun with full schema & telemetry)
  3. [ ] `pilot_llama8b_bf16_math500_seed42` (Llama-8B BF16)
  4. [ ] `pilot_llama8b_fp8_math500_seed42` (Llama-8B FP8 — rerun with full schema & telemetry)
* **Analysis & Endpoints to Compute:**
  * [ ] Pass@1 with 95% Wilson score confidence interval.
  * [ ] Paired discordance contingency table (Both correct, Qwen-only, Llama-only, Both wrong).
  * [ ] McNemar's exact test for paired difference ($p$-value).
  * [ ] Completion token distributions (median, p90, max).
  * [ ] Truncation and true length-cap rate (`finish_reason == "length"`).
  * [ ] Peak VRAM usage and per-token generation latency.
* **Gate P1 Criteria:** All 4 cells pass `integrity_pass`; manual audit of every flagged trace; differences documented regardless of direction.

---

### [ ] Phase 2: Discriminating 3-Seed Pilot Across 4 Formats (24 Cells, Seeds 42, 43, 44)
* **Goal:** Test whether quantization effects replicate across multiple seeds and whether format ordering (BF16 > FP8 > AWQ4 > GPTQ4) is consistent.
* **Task:** `HuggingFaceH4/MATH-500` ($n=500$)
* **Seeds:** `42`, `43`, `44`
* **Formats:** `BF16`, `FP8`, `AWQ4`, `GPTQ4`
* **Cell Matrix (24 Jobs):**
  * **Qwen-7B (12 cells):**
    * [ ] BF16: Seed 42 (from Phase 1), Seed 43, Seed 44
    * [ ] FP8: Seed 42 (from Phase 1), Seed 43, Seed 44
    * [ ] AWQ4: Seed 42, Seed 43, Seed 44
    * [ ] GPTQ4: Seed 42, Seed 43, Seed 44
  * **Llama-8B (12 cells):**
    * [ ] BF16: Seed 42 (from Phase 1), Seed 43, Seed 44
    * [ ] FP8: Seed 42 (from Phase 1), Seed 43, Seed 44
    * [ ] AWQ4: Seed 42, Seed 43, Seed 44
    * [ ] GPTQ4: Seed 42, Seed 43, Seed 44
* **Sample Consistency & Calibration:**
  * [ ] Run `maj@5` (5 completions per item) on a predeclared stratified subset of 100 problems to calculate sample-consistency confidence, Brier score, ECE, and AURC.
* **Gate P2 Criteria:** Produce blinded comparative tables and determine if quantization reliably causes increased truncation, calibration drift, or cost inflation across seeds.

---

### [ ] Phase 3: Primary Contribution Selection Gate
* **Goal:** Select the primary focus and narrative of Paper 1 based on empirical evidence from Phase 2.

| Observed Signal in Phase 2 | Decision Path | Action Required |
|---|---|---|
| Quantization causes statistically significant, reproducible degradation in reliability, calibration, or cost-per-correct across seeds | **Track A: Quantization Reliability–Cost Frontier** | Proceed with 5-seed confirmatory grid and breadth tasks (GPQA, GSM8K). |
| Quantization impact is minimal, but software stack version (0.7.0 vs 0.8.5) and decoding configurations dominate behavior | **Track B: Controlled Serving-Stack Transfer Study** | Build an isolated stack ladder (testing engine version, prompt format, scheduling, and kernels one factor at a time). |
| Neither effect is statistically significant or reproducible | **Negative Results / Replication Artifact** | Terminate GPU expenditure; publish a focused replication report and re-scope with supervisor. |

* **Gate P3 Criteria:** Supervisor alignment and approval on the selected Research Questions and target venue before scheduling any further compute.

---

### [ ] Phase 4: Confirmatory Grid (Headline 5 Seeds)
* **Goal:** Generate publication-grade statistics with 5 seeds (42, 43, 44, 45, 46) for all headline claims.
* [ ] Complete Seeds 45 and 46 for all 8 core model-format combinations on MATH-500 (16 additional cells $\rightarrow$ 40 cells total).
* [ ] If approved by Phase 3 Gate, execute breadth evaluations:
  * [ ] `GPQA-Diamond` ($n=198$, zero-shot, 4 formats $\times$ 3 seeds)
  * [ ] `GSM8K` ($n=1319$, zero-shot, 4 formats $\times$ 3 seeds)
* **Gate P4 Criteria:** Every cell has complete 500/500 rows, verified checksums, and zero uncheckpointed data.

---

### [ ] Phase 5: Frozen Statistical Analysis & Trace Audit
* [ ] **Statistical Testing:**
  * [ ] Paired McNemar tests and paired bootstrap confidence intervals over problems.
  * [ ] Holm-Bonferroni correction across the family of hypothesis tests.
  * [ ] Report seed-to-seed variance separately from problem sampling error.
* [ ] **Economics & Systems:**
  * [ ] Calculate Cost-of-Pass ($C_{\text{pass}}$ / cost-per-correct) under explicit GPU cloud pricing models ($/GPU-hr).
  * [ ] Compute throughput (tok/s), latency percentiles (p50, p95, p99), and peak VRAM.
* [ ] **Trace Audit:**
  * [ ] Perform structured manual review of $\ge 200$ stratified completions.
  * [ ] Audit 100% of cases flagged for length truncation or repetition loops.
* **Gate P5 Criteria:** Reproducible analysis scripts generate all tables and figures directly from raw JSONL files.

---

### [ ] Phase 6: Manuscript Completion & Release Manifest
* [ ] Populate [`paper/main.md`](file:///scratch/manishn_iitp/reasoning-compression-lab/paper/main.md) with finalized tables, figures, and statistical tests.
* [ ] Add comprehensive **Limitations Section** (covering A100 W8A16 Marlin fallback, single-family model scope, and shared-cluster limits).
* [ ] Prepare the open-source release bundle:
  * Pinned requirements & conda lockfiles
  * Patch series in `patches/`
  * Raw execution manifests and configuration hashes
  * Evaluation & scoring scripts
* **Gate P6 Criteria:** Full supervisor sign-off; manuscript is submission-ready.

---

## 5. Summary Progress Tracker

| Stage | Milestone | Status | Blockers / Next Steps |
|---|---|---|---|
| **Phase 0** | Patch creation & clean clone verification | ⏳ In Progress | Create `patches/` and update setup scripts |
| **Phase 0** | Schema extension (`finish_reason`, tokens, timing) | ⏳ Ready to Code | Update inference runner output format |
| **Phase 0** | 3-tier validation & pathology detectors | ⏳ Ready to Code | Update validation scripts |
| **Phase 0** | 3-question GPU smoke test | ⏸️ Pending | Blocked by schema & patch updates |
| **Phase 1** | Matched BF16 vs FP8 (4 cells, seed 42) | ⏸️ Pending | Blocked by Phase 0 |
| **Phase 2** | 3-seed pilot (24 cells, seeds 42–44) | ⏸️ Pending | Blocked by Phase 1 |
| **Phase 3** | Novelty selection gate (Track A vs B) | ⏸️ Pending | Blocked by Phase 2 |
| **Phase 4** | 5-seed confirmatory grid | ⏸️ Pending | Blocked by Phase 3 |
| **Phase 5** | Frozen statistical analysis & audit | ⏸️ Pending | Blocked by Phase 4 |
| **Phase 6** | Final manuscript & artifact release | ⏸️ Pending | Blocked by Phase 5 |
