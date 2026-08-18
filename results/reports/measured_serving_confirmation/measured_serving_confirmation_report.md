# MEASURED SERVING CONFIRMATION BENCHMARK REPORT
**Cluster:** PARAM Rudra HPC (NVIDIA A100-PCIE-80GB)  
**Serving Stack:** `qrm-official` (vLLM 0.7.0 eager, PyTorch 2.5.1+cu124, CUDA 12.4)  
**Pricing Baseline:** $1.50 / A100 GPU-Hour ($0.00041667 / GPU-sec)  

## 1. Executive Summary Table (hybrid scenario Cost-of-Pass)

| Model | Format | Pass@1 | A tok/s | A GPU-s/q | A hybrid $C_{pass}$ [95% CI] | B tok/s | B GPU-s/q | B hybrid $C_{pass}$ [95% CI] | B $\Delta$% vs BF16 [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen-7B** | **BF16** | 94.00% | 43.9±0.0 | 102.51 | $0.0454 [0.0447,0.0463] | 252.7±0.6 | 13.44 | $0.0060 [0.0059,0.0061] | anchor |
| **Qwen-7B** | **FP8** | 94.40% | 62.5±0.1 | 84.60 | $0.0373 [0.0367,0.0380] | 449.8±97.5 | 8.64 | $0.0038 [0.0031,0.0045] | -36.0% [-47.2,-24.8] |
| **Qwen-7B** | **AWQ-4** | 93.12% | 76.9±0.7 | 48.04 | $0.0215 [0.0211,0.0219] | 418.2±2.0 | 7.82 | $0.0035 [0.0034,0.0036] | -41.3% [-41.9,-40.6] |
| **Qwen-7B** | **GPTQ-4** | 93.48% | 82.7±0.7 | 54.39 | $0.0242 [0.0238,0.0248] | 488.3±0.6 | 7.23 | $0.0032 [0.0032,0.0033] | -45.9% [-46.4,-45.4] |
| **Llama-8B** | **BF16** | 89.24% | 72.3±0.3 | 66.43 | $0.0310 [0.0303,0.0318] | 367.0±1.9 | 10.13 | $0.0047 [0.0046,0.0048] | anchor |
| **Llama-8B** | **FP8** | 89.52% | 78.8±2.9 | 92.55 | $0.0431 [0.0415,0.0450] | 481.4±1.4 | 9.32 | $0.0043 [0.0042,0.0044] | -8.2% [-9.5,-6.9] |
| **Llama-8B** | **AWQ-4** | 86.48% | 70.2±2.7 | 89.51 | $0.0431 [0.0415,0.0450] | 391.1±0.5 | 11.63 | $0.0056 [0.0055,0.0058] | +18.5% [+16.6,+20.5] |
| **Llama-8B** | **GPTQ-4** | 88.92% | 63.5±0.4 | 86.95 | $0.0407 [0.0398,0.0418] | 366.2±0.8 | 12.60 | $0.0059 [0.0058,0.0060] | +24.9% [+23.0,+26.7] |

Reported $\pm$ is **sample SD** (`statistics.stdev`, $n-1$). Hybrid scenario Cost-of-Pass uses confirmation GPU-sec in the numerator and campaign MATH-500 pass@1 in the denominator at $1.50/A100-h$. Intervals are Monte Carlo 95% (timing-rep × clustered pass@1). Llama GPTQ-4 mean throughput is about 0.2% lower than Llama BF16; do not say statistically tied.

## 1b. Ranking disagreement (token-proxy vs Condition A vs Condition B)

**Qwen-7B:** rankings disagree across token-proxy, Condition A, and Condition B

| Format | A tok/s rank | A $C_{pass}$ rank | B tok/s rank | B $C_{pass}$ rank | 65 tok/s proxy $C_{pass}$ rank |
|---|---|---|---|---|---|
| BF16 | 4 | 4 | 4 | 4 | 2 |
| FP8 | 3 | 3 | 2 | 3 | 1 |
| AWQ-4 | 2 | 1 | 3 | 2 | 3 |
| GPTQ-4 | 1 | 2 | 1 | 1 | 4 |

**Llama-8B:** rankings disagree across token-proxy, Condition A, and Condition B

| Format | A tok/s rank | A $C_{pass}$ rank | B tok/s rank | B $C_{pass}$ rank | 65 tok/s proxy $C_{pass}$ rank |
|---|---|---|---|---|---|
| BF16 | 2 | 1 | 3 | 2 | 2 |
| FP8 | 1 | 3 | 1 | 1 | 1 |
| AWQ-4 | 3 | 4 | 2 | 3 | 4 |
| GPTQ-4 | 4 | 2 | 4 | 4 | 3 |

## 1c. Qwen-7B FP8 Condition B (keep all five repeats)

Mean 449.79 ± 97.53 tok/s; median 455.90; IQR [350.91, 545.18].

| Rep | tok/s | GPU-s/q | hybrid $C_{pass}$ | regime |
|---|---|---|---|---|
| 1 | 350.91 | 10.6522 | $0.0047 | slow |
| 2 | 350.68 | 10.6591 | $0.0047 | slow |
| 3 | 455.90 | 8.1990 | $0.0036 | mid |
| 4 | 545.18 | 6.8563 | $0.0030 | fast |
| 5 | 546.26 | 6.8427 | $0.0030 | fast |

Slow regime (~351 tok/s, n=2): $C_{pass}$ $0.0047. Mid regime (~456 tok/s, n=1): $C_{pass}$ $0.0036. Fast regime (~546 tok/s, n=2): $C_{pass}$ $0.0030.

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
