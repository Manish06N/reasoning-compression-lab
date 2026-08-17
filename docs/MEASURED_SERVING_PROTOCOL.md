# Measured Serving Systems Benchmark Protocol

**Status:** Frozen on 2026-08-16 *before* executing systems benchmark measurements.  
**Objective:** Empirically measure the real serving throughput (tok/s, req/s), latency distribution (mean, median, P90, P95), peak VRAM, and GPU-seconds per query across 8 precision configurations on NVIDIA A100 GPUs, isolating whether compression efficiency overcomes reasoning token inflation.

---

## 1. Experimental Scope & Model Matrix

The benchmark evaluates 8 canonical configurations across 2 model families:

| ID | Model Family | Checkpoint Format | Serving Engine / Kernel | Precision (`dtype`) |
|---|---|---|---|---|
| **Q1** | Qwen-7B | BF16 (Uncompressed) | vLLM 0.7.0 / PyTorch eager | `bfloat16` |
| **Q2** | Qwen-7B | FP8 (W8A8 checkpoint) | vLLM 0.7.0 / Marlin W8A16 fallback | `bfloat16` |
| **Q3** | Qwen-7B | AWQ-4 (W4A16) | vLLM 0.7.0 / AWQ Marlin kernel | `float16` |
| **Q4** | Qwen-7B | GPTQ-4 (W4A16) | vLLM 0.7.0 / GPTQ Marlin kernel | `float16` |
| **L1** | Llama-8B | BF16 (Uncompressed) | vLLM 0.7.0 / PyTorch eager | `bfloat16` |
| **L2** | Llama-8B | FP8 (W8A8 checkpoint) | vLLM 0.7.0 / Marlin W8A16 fallback | `bfloat16` |
| **L3** | Llama-8B | AWQ-4 (W4A16) | vLLM 0.7.0 / AWQ Marlin kernel | `float16` |
| **L4** | Llama-8B | GPTQ-4 (W4A16) | vLLM 0.7.0 / GPTQ Marlin kernel | `float16` |

---

## 2. Hardware & Serving Stack Specifications

All benchmark jobs run on dedicated NVIDIA A100-PCIE-80GB GPUs on PARAM Rudra HPC under identical environment controls:
- **GPU Resource:** 1× A100 80GB GPU per cell (`--gres=gpu:1`, `--cpus-per-task=16`).
- **Serving Engine:** vLLM 0.7.0 in `qrm-official` conda environment (`/home/manishn_iitp/.conda/envs/qrm-official`).
- **Toolchain:** PyTorch 2.5.1+cu124, Transformers 4.47.1, CUDA 12.4, GCC 12.
- **Engine Arguments:**
  - `enforce_eager=True` (required for compute node stability).
  - `gpu_memory_utilization=0.75` (reserves 60GB on 80GB A100).
  - `max_model_len=32768`.
  - `tensor_parallel_size=1`.
- **Sampling Parameters:**
  - `temperature=0.6`, `top_p=0.95`, `repetition_penalty=1.0`, `max_new_tokens=32768`.
  - Sampling seed pinned to `20260816`.

---

## 3. Representative Input Workload Selection

- **Dataset:** `HuggingFaceH4/MATH-500` ($n=500$).
- **Subset Size:** 100 problems ($n=100$) stratified across all 5 difficulty levels (20 problems each from Levels 1, 2, 3, 4, and 5).
- **Selection Seed:** Deterministic pseudo-random seed `20260816`.
- **Integrity:** The identical prompt list and ordering are frozen in `results/measured_serving/input_subset.json` and served to all 8 configurations.

---

## 4. Serving Conditions

Each configuration is evaluated under two distinct serving regimes:

### Condition A: Low-Concurrency / Interactive ($C=1$)
- **Concurrency:** 1 request at a time (sequential stream).
- **Workload:** 20 stratified prompts (4 from each difficulty Level 1–5) drawn from the frozen input subset.
- **Repetitions:** 3 independent runs ($R=3$).
- **Primary Utility:** Measures single-user interactive generation latency, time-per-output-token, median/P90/P95 latency, and single-stream decode speed.

### Condition B: Batched Throughput ($C=8$)
- **Concurrency:** Fixed continuous batching concurrency of 8 parallel asynchronous requests.
- **Workload:** Full 100 stratified prompts (20 from each difficulty Level 1–5) drawn from the frozen input subset.
- **Repetitions:** 3 independent runs ($R=3$).
- **Primary Utility:** Measures multi-tenant serving throughput, batch decoding efficiency, and hardware saturation under shared memory constraints.

### Secondary Microbenchmark: Fixed-Token Decoding
- **Workload:** 10 prompts evaluated with forced generation of exactly 512 tokens (`min_tokens=512`, `max_tokens=512`, `ignore_eos=True`, $T=0.0$).
- **Primary Utility:** Isolates raw hardware/kernel decoding speed from reasoning trace length variations.

---

## 5. Warmup & Repetitions

1. **Warmup Phase:**
   - Immediately following engine initialization, 3 dedicated warmup queries (drawn outside the benchmark subset) are processed.
   - All warmup latency, token counts, and execution times are strictly discarded from steady-state measurements.
2. **Repetition Count:**
   - Every `(configuration, serving_condition)` pair is executed across **3 independent repeated runs** ($R=3$).
   - Node hostname, GPU device ID, and process metadata are captured for each repetition.

---

## 6. Metric Definitions & Accounting Rules

1. **Output Throughput ($T_{\text{out}}$):**
   $$\text{Output tok/s} = \frac{\sum_{i=1}^{N} \text{output\_tokens}_i}{\Delta t_{\text{steady}}}$$
   where $\Delta t_{\text{steady}}$ is the elapsed steady-state wall-clock time between the launch of the first measured request and the return of the final response.
2. **Request Throughput ($R_{\text{req}}$):**
   $$\text{Requests/s} = \frac{N}{\Delta t_{\text{steady}}}$$
3. **Per-Request Latency:**
   - Latency per request $L_i = t_{\text{end}, i} - t_{\text{start}, i}$.
   - Reported as Mean, Median, P90, and P95.
4. **Memory Footprint:**
   - **Model Loaded VRAM:** GPU allocated memory immediately after weights are loaded into VRAM.
   - **Peak VRAM:** Maximum GPU memory allocated and reserved during active batch generation (`torch.cuda.max_memory_allocated()`).
5. **Measured GPU-Seconds per Query:**
   $$\text{GPU-sec / query} = \frac{\Delta t_{\text{steady}} \times N_{\text{GPU}}}{N_{\text{completed}}}$$
6. **Measured Cost-of-Pass ($C_{\text{pass}}^{\text{meas}}$):**
   Under the standard cloud datacenter scenario of $\$1.50 / \text{A100 GPU-hour}$ ($\$0.0004167 / \text{GPU-sec}$):
   $$\text{Cost / query} = (\text{GPU-sec / query}) \times \left(\frac{\$1.50}{3600}\right)$$
   $$C_{\text{pass}}^{\text{meas}} = \frac{\text{Cost / query}}{\text{Pass@1}}$$
   where $\text{Pass@1}$ is the frozen canonical accuracy from the 40-cell MATH-500 campaign.

---

## 7. Implementation notes after execution (do not rewrite the frozen design above)

These are observed differences between the frozen protocol and `scripts/hpc/qrm_parity/benchmark_serving.py` as run:

- Condition B submits all 100 prompts in one `llm.generate` call. The JSON field `concurrency` is labeled `8`, but `LLM(..., max_num_seqs=8)` was **not** set (vLLM 0.7.0 default continuous batching).
- Condition A uses `prompts_all[:20]`. The frozen 100-item list is 20 problems per MATH level; the first 20 are **not** 4-per-level (observed counts 5/7/3/3/2).
- `results/measured_serving/input_subset.json` `problem`/`solution` sidecar fields are misaligned with `full_prompt`. Serving used `full_prompt`.
- Sampling seed `20260816` is shared across the three repeats, so output token counts are identical; $R=3$ measures wall-clock noise.
- Peak VRAM is `torch.cuda.max_memory_allocated()` after `gpu_memory_utilization=0.75` preallocation, not isolated weight footprint.
- Llama AWQ-4 and Llama GPTQ-4 raw files include more than one hostname (cache reuse / re-execution).
- Do not cite a unique “true Pareto optimum.” Compute dominance on stated objectives.
