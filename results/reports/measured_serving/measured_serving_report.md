# Measured Serving Performance & Cost-of-Pass Systems Benchmark Report

**Hardware:** NVIDIA A100-PCIE-80GB (PARAM Rudra HPC)  
**Serving Engine:** vLLM 0.7.0 eager (`qrm-official` conda env) | **Toolchain:** PyTorch 2.5.1+cu124, CUDA 12.4  
**Dataset:** MATH-500 stratified benchmark subset ($n=100$) | **Repetitions:** $R=3$ wall-clock repeats (shared sampling seed)  
**Pricing Baseline:** $\$1.50 / \text{A100 GPU-hour}$ ($\$0.00041667 / \text{GPU-second}$)  

---

## 1. Measured Serving Performance Across Conditions

### Condition A: Low-Concurrency / Interactive Stream ($C=1$)

| Model & Format | Output tok/s | Median Latency (s) | P90 Latency (s) | Peak VRAM (GB) | GPU-sec / query | Measured $C_{\text{pass}}$ ($) |
|---|---|---|---|---|---|---|
| **Qwen-7B BF16** | 79.91 ± 0.07 | 32.97 | 77.76 | 54.97 | 45.44 | $0.0201 |
| **Qwen-7B FP8** | 85.89 ± 0.79 | 41.87 | 69.14 | 54.98 | 46.07 | $0.0203 |
| **Qwen-7B AWQ-4** | 72.76 ± 0.16 | 51.47 | 174.46 | 55.08 | 73.96 | $0.0331 |
| **Qwen-7B GPTQ-4** | 68.34 ± 0.66 | 46.20 | 81.85 | 53.87 | 48.38 | $0.0216 |
| **Llama-8B BF16** | 73.41 ± 0.70 | 47.66 | 192.52 | 55.92 | 74.79 | $0.0349 |
| **Llama-8B FP8** | 80.44 ± 1.22 | 40.23 | 113.75 | 55.93 | 60.87 | $0.0283 |
| **Llama-8B AWQ-4** | 69.43 ± 2.09 | 49.00 | 212.65 | 56.13 | 79.97 | $0.0385 |
| **Llama-8B GPTQ-4** | 71.60 ± 0.96 | 59.53 | 227.90 | 55.99 | 99.00 | $0.0464 |

### Condition B: Batched Throughput ($C=8$)

| Model & Format | Output tok/s | Requests/s | Peak VRAM (GB) | GPU-sec / query | Cost / query ($) | Measured $C_{\text{pass}}$ ($) |
|---|---|---|---|---|---|---|
| **Qwen-7B BF16** | 694.27 ± 0.65 | 0.178 | 56.00 | 5.62 | $0.0023 | $0.0025 |
| **Qwen-7B FP8** | 824.27 ± 4.00 | 0.221 | 56.02 | 4.52 | $0.0019 | $0.0020 |
| **Qwen-7B AWQ-4** | 687.51 ± 0.59 | 0.170 | 56.00 | 5.89 | $0.0025 | $0.0026 |
| **Qwen-7B GPTQ-4** | 602.65 ± 1.40 | 0.149 | 55.17 | 6.70 | $0.0028 | $0.0030 |
| **Llama-8B BF16** | 649.72 ± 0.51 | 0.146 | 56.71 | 6.86 | $0.0029 | $0.0032 |
| **Llama-8B FP8** | 759.48 ± 6.96 | 0.157 | 56.72 | 6.37 | $0.0027 | $0.0030 |
| **Llama-8B AWQ-4** | 545.68 ± 125.62 | 0.105 | 56.72 | 10.15 | $0.0042 | $0.0049 |
| **Llama-8B GPTQ-4** | 785.55 ± 12.33 | 0.152 | 56.74 | 6.58 | $0.0027 | $0.0031 |

---

## 2. Relative Systems Deltas vs BF16 Anchors ($\Delta = \text{Quantized} - \text{BF16}$)

| Configuration vs BF16 | Condition | $\Delta$ Output tok/s | $\Delta$ GPU-sec/query | $\Delta$ Peak VRAM | $\Delta$ Cost-of-Pass ($C_{\text{pass}}$) |
|---|---|---|---|---|---|
| **Qwen-7B FP8 vs BF16** | Interactive (C=1) | +7.5% | +1.4% | +0.0% | +1.0% |
| **Qwen-7B FP8 vs BF16** | Batched (C=8) | +18.7% | -19.5% | +0.0% | -19.8% |
| **Qwen-7B AWQ-4 vs BF16** | Interactive (C=1) | -8.9% | +62.8% | +0.2% | +64.3% |
| **Qwen-7B AWQ-4 vs BF16** | Batched (C=8) | -1.0% | +4.8% | +0.0% | +5.8% |
| **Qwen-7B GPTQ-4 vs BF16** | Interactive (C=1) | -14.5% | +6.5% | -2.0% | +7.1% |
| **Qwen-7B GPTQ-4 vs BF16** | Batched (C=8) | -13.2% | +19.3% | -1.5% | +20.0% |
| **Llama-8B FP8 vs BF16** | Interactive (C=1) | +9.6% | -18.6% | +0.0% | -18.9% |
| **Llama-8B FP8 vs BF16** | Batched (C=8) | +16.9% | -7.1% | +0.0% | -7.4% |
| **Llama-8B AWQ-4 vs BF16** | Interactive (C=1) | -5.4% | +6.9% | +0.4% | +10.3% |
| **Llama-8B AWQ-4 vs BF16** | Batched (C=8) | -16.0% | +47.8% | +0.0% | +52.5% |
| **Llama-8B GPTQ-4 vs BF16** | Interactive (C=1) | -2.5% | +32.4% | +0.1% | +32.8% |
| **Llama-8B GPTQ-4 vs BF16** | Batched (C=8) | +20.9% | -4.1% | +0.1% | -3.8% |

---

## 3. Comparison: Old Fixed-Throughput Proxy vs New Measured Serving Cost

| Model & Format | Pass@1 (Accuracy) | Old Proxy Cost/Query | Old Proxy $C_{\text{pass}}$ | Measured Batched Cost/Query | Measured Batched $C_{\text{pass}}$ | (pass@1, $C_{\text{pass}}$) |
|---|---|---|---|---|---|---|
| **Qwen-7B BF16** | 94.00% | $0.0243 | $0.0258 | $0.0023 | $0.0025 | dominated on (pass@1, C_pass) |
| **Qwen-7B FP8** | 94.40% | $0.0243 | $0.0257 | $0.0019 | $0.0020 | nondominated (pass@1, C_pass) |
| **Qwen-7B AWQ-4** | 93.12% | $0.0258 | $0.0277 | $0.0025 | $0.0026 | dominated on (pass@1, C_pass) |
| **Qwen-7B GPTQ-4** | 93.48% | $0.0260 | $0.0278 | $0.0028 | $0.0030 | dominated on (pass@1, C_pass) |
| **Llama-8B BF16** | 89.24% | $0.0285 | $0.0319 | $0.0029 | $0.0032 | dominated on (pass@1, C_pass) |
| **Llama-8B FP8** | 89.52% | $0.0279 | $0.0311 | $0.0027 | $0.0030 | dominated on (pass@1, C_pass) |
| **Llama-8B AWQ-4** | 86.48% | $0.0290 | $0.0335 | $0.0042 | $0.0049 | dominated on (pass@1, C_pass) |
| **Llama-8B GPTQ-4** | 88.92% | $0.0296 | $0.0333 | $0.0027 | $0.0031 | dominated on (pass@1, C_pass) |

### Pareto note

There is no unique ``true Pareto optimum.'' On batched (pass@1, measured $C_{\text{pass}}$) the nondominated pooled set is: Qwen-7B FP8.
Qwen FP8 is Pareto-efficient in that two-objective set; Qwen GPTQ-4 is dominated. $1.50$/A100-h is a pricing scenario. Pass@1 is the 40-cell MATH-500 campaign mean.
