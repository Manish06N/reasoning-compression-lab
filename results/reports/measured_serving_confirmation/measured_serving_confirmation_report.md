# MEASURED SERVING CONFIRMATION BENCHMARK REPORT
**Cluster:** PARAM Rudra HPC (NVIDIA A100-PCIE-80GB)  
**Serving Stack:** `qrm-official` (vLLM 0.7.0 eager, PyTorch 2.5.1+cu124, CUDA 12.4)  
**Pricing Baseline:** $1.50 / A100 GPU-Hour ($0.00041667 / GPU-sec)  

## 1. Executive Summary Table

| Model | Format | Pass@1 (MATH-500) | Cond A Tok/s (C=1) | Cond A Median Lat (s) | Cond B Tok/s (C=8) | Cond B Req/s | Empirical GPU-sec/q | Measured $C_{\text{pass}}$ | $C_{\text{pass}}$ Delta vs BF16 |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen-7B** | **BF16** | 94.00% | 43.9 ± 0.0 | 59.19s | 252.7 ± 0.6 | 0.074 | 13.44s | $0.0060 | Anchor |
| **Qwen-7B** | **FP8** | 94.40% | 62.5 ± 0.1 | 43.60s | 449.8 ± 97.5 | 0.120 | 8.64s | $0.0038 | -36.0% |
| **Qwen-7B** | **AWQ-4** | 93.12% | 76.9 ± 0.7 | 26.47s | 418.2 ± 2.0 | 0.128 | 7.82s | $0.0035 | -41.3% |
| **Qwen-7B** | **GPTQ-4** | 93.48% | 82.7 ± 0.7 | 34.78s | 488.3 ± 0.6 | 0.138 | 7.23s | $0.0032 | -45.9% |
| **Llama-8B** | **BF16** | 89.24% | 72.3 ± 0.3 | 32.40s | 367.0 ± 1.9 | 0.099 | 10.13s | $0.0047 | Anchor |
| **Llama-8B** | **FP8** | 89.52% | 78.8 ± 2.9 | 38.00s | 481.4 ± 1.4 | 0.107 | 9.32s | $0.0043 | -8.2% |
| **Llama-8B** | **AWQ-4** | 86.48% | 70.2 ± 2.7 | 40.12s | 391.1 ± 0.5 | 0.086 | 11.63s | $0.0056 | +18.5% |
| **Llama-8B** | **GPTQ-4** | 88.92% | 63.5 ± 0.4 | 46.31s | 366.2 ± 0.8 | 0.079 | 12.60s | $0.0059 | +24.9% |

Reported $\pm$ is **sample SD** (`statistics.stdev`, $n-1$). The runner's CV-expansion trigger uses **population SD** (`np.std`, $n$ divisor) on the first three repeats. Do not call the trigger statistic a sample SD. Cost-of-Pass is **scenario-based** at $\$1.50$/A100-hour, not billed cluster cost.

## 2. Secondary Fixed-Token Microbenchmark (Pure Decode Speed)

| Model | Format | Fixed Tokens | Raw Decode Tok/s | Speedup vs BF16 |
|---|---|---|---|---|
| **Qwen-7B** | **BF16** | 512 | 205.41 tok/s | 1.00× |
| **Qwen-7B** | **FP8** | 512 | 437.19 tok/s | 2.13× |
| **Qwen-7B** | **AWQ-4** | 512 | 377.73 tok/s | 1.84× |
| **Qwen-7B** | **GPTQ-4** | 512 | 401.19 tok/s | 1.95× |
| **Llama-8B** | **BF16** | 512 | 362.01 tok/s | 1.00× |
| **Llama-8B** | **FP8** | 512 | 392.05 tok/s | 1.08× |
| **Llama-8B** | **AWQ-4** | 512 | 344.04 tok/s | 0.95× |
| **Llama-8B** | **GPTQ-4** | 512 | 302.54 tok/s | 0.84× |
