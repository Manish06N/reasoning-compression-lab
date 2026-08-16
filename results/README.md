# Experimental Results & Validation Archive
**Repository:** reasoning-compression-lab  
**Cluster:** PARAM Rudra HPC (NVIDIA A100 80GB GPUs)  
**Last Synced:** 2026-08-16 19:30:42  

This directory contains the verified, reproducible evaluation results across all quantization formats (BF16, FP8, AWQ-4, GPTQ-4), architectures (Qwen-7B, Llama-8B), and random seeds.

---

## 1. Benchmark Summary Matrices

### MATH-500 (n=500, 5 Seeds) Evaluation Matrix

| Model Family | Format | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 | Mean ± Std |
|---|---|---|---|---|---|---|---|
| Qwen-7B | BF16 | 94.40% | 94.00% | 93.80% | 94.60% | 93.20% | **94.00% ± 0.55%** |
| Qwen-7B | FP8 | 94.40% | 95.20% | 94.80% | 92.60% | 95.00% | **94.40% ± 1.05%** |
| Qwen-7B | AWQ-4 | 92.40% | 92.80% | 93.20% | 93.00% | 94.20% | **93.12% ± 0.67%** |
| Qwen-7B | GPTQ-4 | 93.80% | 92.60% | 93.40% | 94.60% | 93.00% | **93.48% ± 0.77%** |
| Llama-8B | BF16 | 89.00% | 88.40% | 90.20% | 89.80% | 88.80% | **89.24% ± 0.74%** |
| Llama-8B | FP8 | 89.00% | 89.60% | 88.60% | 89.20% | 91.20% | **89.52% ± 1.01%** |
| Llama-8B | AWQ-4 | 84.40% | 84.80% | 89.20% | 87.40% | 86.60% | **86.48% ± 1.96%** |
| Llama-8B | GPTQ-4 | 88.00% | 89.60% | 86.80% | 89.40% | 90.80% | **88.92% ± 1.55%** |

---

### GSM8K (n=1,319, 3 Seeds) Evaluation Matrix

| Model Family | Format | Seed 42 | Seed 43 | Seed 44 | Mean ± Std |
|---|---|---|---|---|---|
| Qwen-7B | BF16 | 91.05% | 91.58% | 91.13% | **91.26% ± 0.29%** |
| Qwen-7B | FP8 | 91.28% | 91.51% | 91.21% | **91.33% ± 0.16%** |
| Qwen-7B | AWQ-4 | 91.05% | 89.92% | 92.19% | **91.05% ± 1.14%** |
| Qwen-7B | GPTQ-4 | 90.90% | 91.43% | 91.05% | **91.13% ± 0.27%** |
| Llama-8B | BF16 | 88.17% | 88.78% | 89.08% | **88.68% ± 0.46%** |
| Llama-8B | FP8 | 89.08% | 89.23% | 88.10% | **88.80% ± 0.62%** |
| Llama-8B | AWQ-4 | 87.34% | 86.88% | 87.11% | **87.11% ± 0.23%** |
| Llama-8B | GPTQ-4 | 88.48% | 88.63% | 89.76% | **88.96% ± 0.70%** |

---

### GPQA-Diamond (n=198, 3 Seeds) Evaluation Matrix

| Model Family | Format | Seed 42 | Seed 43 | Seed 44 | Mean ± Std |
|---|---|---|---|---|---|
| Qwen-7B | BF16 | 51.52% | 46.97% | 52.53% | **50.34% ± 2.96%** |
| Qwen-7B | FP8 | 49.49% | 51.01% | 47.98% | **49.49% ± 1.52%** |
| Qwen-7B | AWQ-4 | 44.44% | 41.92% | 47.98% | **44.78% ± 3.04%** |
| Qwen-7B | GPTQ-4 | 46.97% | 50.00% | 46.97% | **47.98% ± 1.75%** |
| Llama-8B | BF16 | 43.94% | 46.97% | 47.47% | **46.13% ± 1.91%** |
| Llama-8B | FP8 | 47.47% | 47.98% | 47.98% | **47.81% ± 0.29%** |
| Llama-8B | AWQ-4 | 46.97% | 44.95% | 48.99% | **46.97% ± 2.02%** |
| Llama-8B | GPTQ-4 | 44.44% | 40.91% | 49.49% | **44.95% ± 4.32%** |

---

## 2. Directory Structure

```
results/
├── math500/      # 40 official validation records (5 seeds × 4 formats × 2 models)
├── gsm8k/        # 24 breadth validation records (3 seeds × 4 formats × 2 models)
├── gpqa/         # 24 breadth validation records (3 seeds × 4 formats × 2 models)
├── reports/      # Statistical analysis, calibration ECE, and trace audit reports
└── README.md     # Consolidated summary manifest and score tables
```

## 3. Data Integrity & Verification
- All results generated on PARAM Rudra HPC 2× A100 GPUs under `qrm-official` (vLLM 0.7.0 eager).
- Decoding parameters: $T=0.6, p=0.95, \text{max\_tokens}=32,768$.
- Pathological degeneration rate: **0 truncations, 0 repetition loops** across all cells under the **heuristic** detectors (encoded length $\ge 32{,}768$; consecutive identical-word runs). Official QRM rows do not store vLLM `finish_reason`.
