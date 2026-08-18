# MEASURED_SERVING_CONFIRMATION_PROTOCOL.md — Controlled Systems Confirmation Benchmark

**Author:** Manish Nandish, IIT Patna  
**Target Venue:** *Future Generation Computer Systems (FGCS)* / *Journal of Systems and Software (JSS)*  
**Repository:** `/scratch/manishn_iitp/reasoning-compression-lab`  
**Date:** 2026-08-17 (Pre-Rerun Frozen Protocol)  
**Status:** **Executed and frozen.** Preferred serving numbers are in `results/reports/measured_serving_confirmation/`. Do not re-run. GPU work is closed.  
**Execution Mode:** Confirmatory Verification Grid under Strict Apples-to-Apples Controls  

---

## 1. Scientific Objective & Boundary Conditions

### 1.1 Purpose
The purpose of this benchmark is to **confirm the empirical systems throughput, latency, and Cost-of-Pass findings** of the Paper 1 campaign under stricter, fully audited controls. This is **not an exploratory experiment**; it directly resolves the protocol limitations identified during local audit:
1. **Pinned Concurrency Control in Condition B:** Explicitly pass and verify `max_num_seqs=8` to the vLLM 0.7.0 scheduler, moving from unconstrained batching to pinned concurrency scheduling.
2. **Balanced Stratification in Condition A:** Establish a deterministic, balanced 20-problem subset with exactly 4 items per difficulty level (Levels 1–5) to eliminate difficulty skew.
3. **Full Sidecar Provenance Alignment:** Guarantee 100% bijective alignment between `problem_index`, `difficulty`, `problem`, `solution`, `answer`, and `full_prompt`.
4. **Strict Single-Node Physical Control:** Eliminate inter-node variance by executing all configurations and repetitions per architecture on the same physical NVIDIA A100 node (e.g. `ragpu004` or `ragpu003`), explicitly resolving the contaminated Llama AWQ-4 multi-node outlier.
5. **Technical Replicate Metrology:** Formalize repetitions as technical timing replicates under an identical generated-token workload (shared sampling seed), evaluating coefficient of variation ($\text{CV} \le 3\%$) to govern replicate sufficiency.

### 1.2 Boundary Conditions & Invariants
- **No Changes to Sampling Hyperparameters:** $T=0.6, p=0.95, \text{repetition\_penalty}=1.0, \text{max\_tokens}=32,768$.
- **No Changes to Software Stack:** vLLM 0.7.0 eager (`qrm-official`), PyTorch 2.5.1+cu124, CUDA 12.4, `--gpu-memory-utilization 0.75`, `--enforce-eager`.
- **Accuracy Invariant:** All Cost-of-Pass calculations use the canonical 40-cell MATH-500 campaign pass@1 accuracy numbers (5 seeds, $n=500$). Serving subsets are used exclusively for systems timing.

---

## 2. Benchmark Grid & Configurations (8 Cells)

The confirmation grid evaluates the identical 8 configurations from Paper 1:

| Model Family | Weight Format | Quantization Kernel / Engine Mode | Checkpoint Disk Path |
|---|---|---|---|
| **Qwen-7B** | **BF16** | Native torch.bfloat16 eager | `/scratch/manishn_iitp/reasoning-compression-lab/models/DeepSeek-R1-Distill-Qwen-7B` |
| **Qwen-7B** | **FP8** | Marlin W8A16 weight-only fallback | `/scratch/manishn_iitp/reasoning-compression-lab/models/DeepSeek-R1-Distill-Qwen-7B-FP8` |
| **Qwen-7B** | **AWQ-4** | GEMM AWQ kernel (`--dtype float16`) | `/scratch/manishn_iitp/reasoning-compression-lab/models/DeepSeek-R1-Distill-Qwen-7B-AWQ-4` |
| **Qwen-7B** | **GPTQ-4** | Marlin W4A16 fused tensor kernel (`--dtype float16`) | `/scratch/manishn_iitp/reasoning-compression-lab/models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-4` |
| **Llama-8B** | **BF16** | Native torch.bfloat16 eager | `/scratch/manishn_iitp/reasoning-compression-lab/models/DeepSeek-R1-Distill-Llama-8B` |
| **Llama-8B** | **FP8** | Marlin W8A16 weight-only fallback | `/scratch/manishn_iitp/reasoning-compression-lab/models/DeepSeek-R1-Distill-Llama-8B-FP8` |
| **Llama-8B** | **AWQ-4** | GEMM AWQ kernel (`--dtype float16`) | `/scratch/manishn_iitp/reasoning-compression-lab/models/DeepSeek-R1-Distill-Llama-8B-AWQ-4` |
| **Llama-8B** | **GPTQ-4** | Marlin W4A16 fused tensor kernel (`--dtype float16`) | `/scratch/manishn_iitp/reasoning-compression-lab/models/DeepSeek-R1-Distill-Llama-8B-GPTQ-4` |

---

## 3. Workload Protocol & Conditions

### 3.1 Condition A: Interactive Single-Stream ($C=1$)
- **Dataset:** Frozen 20-item balanced subset ([`results/measured_serving_confirmation/condition_a_subset.json`](file:///scratch/manishn_iitp/reasoning-compression-lab/results/measured_serving_confirmation/condition_a_subset.json)).
- **Stratification:** Exactly **4 problems from Level 1, 4 from Level 2, 4 from Level 3, 4 from Level 4, and 4 from Level 5** selected deterministically via seed `20260817`.
- **Execution:** Sequential single-request generation (`concurrency=1`).
- **Metrics Captured:**
  - Mean & Median Latency (seconds)
  - P90 / P95 Latency (seconds)
  - Output decoding tokens/second
  - Total input and output tokens
  - GPU-seconds per query

### 3.2 Condition B: Pinned Concurrency Batched Throughput ($C=8$)
- **Dataset:** Frozen 100-item balanced subset ([`results/measured_serving_confirmation/condition_b_subset.json`](file:///scratch/manishn_iitp/reasoning-compression-lab/results/measured_serving_confirmation/condition_b_subset.json)), 20 items per Level (1–5).
- **Concurrency Pinning:** Engine initialized with `max_num_seqs = 8` explicitly passed into `LLM(..., max_num_seqs=8)`.
- **Assertion:** Runtime verification that `llm.llm_engine.scheduler_config.max_num_seqs == 8`.
- **Metrics Captured:**
  - Aggregate decoding throughput (tokens/second)
  - Completed query throughput (requests/second)
  - Empirical GPU-seconds per query ($\Delta t_{\text{steady}} / N_{\text{completed}}$)
  - Measured Cost-of-Pass ($C_{\text{pass}}^{\text{meas}}$)

### 3.3 Secondary Microbenchmark: Fixed-Token Pure Decode
- **Workload:** 10 sample prompts generating exactly 512 tokens with `ignore_eos=True` and $T=0.0$.
- **Purpose:** Direct isolation of raw hardware decode speed in the absence of variable reasoning chain termination.

---

## 4. Hardware Metrology, Node Control & Replicates

### 4.1 Physical Node Invariance
- To prevent hardware-node performance anomalies, all four formats of an architecture MUST execute sequentially on the same physical compute node.
- Provenance logs capture `hostname`, `gpu_device_uuid`, `driver_version`, and `cuda_version` for every execution.

### 4.2 Technical Timing Replicates
- **Definition:** Repetitions ($R \ge 3$) share the same sampling seed (`20260816`). This produces identical generated output token sequences across runs, isolating pure serving-system variance (CUDA kernels, memory bandwidth, scheduling latency).
- **Sufficiency Rule:**
  $$\text{CV} = \frac{\sigma_{\text{tok/s}}}{\mu_{\text{tok/s}}} \times 100\%$$
  - If $\text{CV} \le 3.0\%$: 3 replicates are sufficient.
  - If $\text{CV} > 3.0\%$: 2 additional technical replicates are automatically executed ($R=5$).

### 4.3 Warmup & Steady-State Isolation
- 3 warm-up requests are executed and discarded prior to measurement.
- Explicit `torch.cuda.synchronize()` calls bracket all `time.perf_counter()` timestamps.

---

## 5. Cost-of-Pass Formulation

Under the datacenter pricing baseline ($1.50/\text{A100 GPU-hour} = \$0.00041667/\text{GPU-second}$):

$$\text{Cost per query} = (\text{GPU-sec per query}) \times \left( \frac{\$1.50}{3600} \right)$$

$$C_{\text{pass}}^{\text{meas}} = \frac{\text{Cost per query}}{\text{Pass@1}_{\text{canonical}}}$$

---

## 6. Output Structure & Integrity Checks

All confirmatory results are written to isolated directories to preserve legacy records:
- Raw Runs: `results/measured_serving_confirmation/raw/`
- Provenance: `results/reports/measured_serving_confirmation/provenance/`
- Reports:
  - `results/reports/measured_serving_confirmation/measured_serving_confirmation_report.json`
  - `results/reports/measured_serving_confirmation/measured_serving_confirmation_report.md`
  - `results/reports/measured_serving_confirmation/measured_serving_confirmation_validation.md`
