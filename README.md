# Reasoning Compression Lab (`reasoning-compression-lab`)

**Deployment-science evaluation harness and reliability–cost frontier benchmark for compressed reasoning LLMs.**

* **PhD Scholar:** Manish Nandish (IIT Patna)
* **Cluster:** PARAM Rudra HPC (C-DAC / NSM), NVIDIA A100-PCIE-80GB GPUs
* **GitHub:** [https://github.com/Manish06N/reasoning-compression-lab](https://github.com/Manish06N/reasoning-compression-lab)
* **Paper 1 (J1):** *Beyond Pass@1: Reliability–Cost Frontiers of Quantized Reasoning Models under Controlled Serving-Stack Shift*

---

## 1. Executive Summary & Breakthrough Results (August 2026)

This repository contains the official codebase, execution pipelines, statistical analysis tools, and open-source reproducibility artifacts for evaluating post-training quantization (BF16, FP8, AWQ-4, GPTQ-4) on long-form reasoning models (`DeepSeek-R1-Distill-Qwen-7B` and `DeepSeek-R1-Distill-Llama-8B`).

### Headline Confirmatory Accuracy Matrix (MATH-500, $n=500$, Seeds 42–46, 20,000 Completions)
All 40 experimental cells under **Protocol P1-2026-08** (`vLLM 0.7.0` eager execution, $T=0.6, p=0.95, \text{max\_tokens}=32,768$) achieved **0 length truncations** and **0 infinite repetition loops**:

| Model & Format | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 | Mean ± Std | 95% Wilson CI | Truncations | Repetition Loops |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen-7B BF16** | 94.4% | 94.0% | 93.8% | 94.6% | 93.2% | **94.00% ± 0.55%** | [93.0%, 94.9%] | 0 | 0 |
| **Qwen-7B FP8** | 94.4% | 95.2% | 94.8% | 92.6% | 95.0% | **94.40% ± 1.05%** | [93.4%, 95.2%] | 0 | 0 |
| **Qwen-7B AWQ-4** | 92.4% | 92.8% | 93.2% | 93.0% | 94.2% | **93.12% ± 0.67%** | [92.1%, 94.0%] | 0 | 0 |
| **Qwen-7B GPTQ-4** | 93.8% | 92.6% | 93.4% | 94.6% | 93.0% | **93.48% ± 0.77%** | [92.4%, 94.4%] | 0 | 0 |
| **Llama-8B BF16** | 89.0% | 88.4% | 90.2% | 89.8% | 88.8% | **89.24% ± 0.74%** | [88.0%, 90.4%] | 0 | 0 |
| **Llama-8B FP8** | 89.0% | 89.6% | 88.6% | 89.2% | 91.2% | **89.52% ± 1.01%** | [88.3%, 90.7%] | 0 | 0 |
| **Llama-8B AWQ-4** | 84.4% | 84.8% | 89.2% | 87.4% | 86.6% | **86.48% ± 1.96%** | [85.1%, 87.8%] | 0 | 0 |
| **Llama-8B GPTQ-4** | 88.0% | 89.6% | 86.8% | 89.4% | 90.8% | **88.92% ± 1.55%** | [87.6%, 90.1%] | 0 | 0 |

### Breadth Benchmark Results (GSM8K, $n=1,319$, Seeds 42–44)
* **Qwen-7B:** BF16: 91.26% ± 0.23% | FP8: 91.33% ± 0.13% | AWQ-4: 91.05% ± 0.93% | GPTQ-4: 91.13% ± 0.22%
* **Llama-8B:** BF16: 88.68% ± 0.38% | FP8: 88.80% ± 0.50% | AWQ-4: 87.11% ± 0.19% | GPTQ-4: 88.96% ± 0.58%

---

## 2. Key Scientific Findings

1. **FP8 Statistical Parity:** On NVIDIA A100 GPUs (via Marlin W8A16 fallback), FP8 achieves complete statistical parity with full-precision BF16 baselines across both architectures (exact paired McNemar tests show no significant discordance under Holm-Bonferroni correction, $p > 0.05$).
2. **Architecture Quantization Resilience:** Qwen-7B demonstrates superior robustness to 4-bit compression (>93.1% accuracy preserved), whereas Llama-8B exhibits higher sensitivity to 4-bit AWQ compression (86.48% vs 89.24% BF16).
3. **Zero Pathological Degenerations:** Pinned eager execution with a 32,768-token budget eliminated the catastrophic context-cap truncations and repetition loops previously reported in uncalibrated serving configurations.
4. **Token Inflation & Cost-of-Pass ($C_{\text{pass}}$):** 4-bit quantization introduces a $+3.9\%$ to $+6.5\%$ token inflation penalty, shifting the Pareto frontier such that **FP8 consistently delivers the optimal dollar-cost-per-correct answer**.

---

## 3. Repository Structure

```
reasoning-compression-lab/
├── configs/               # Campaign cell configurations (MATH-500, GSM8K, GPQA-Diamond)
├── docs/                  # PhD Roadmap, literature surveys, supervisor briefing, hardware policies
│   ├── literature/        # Reading maps and base paper summaries
│   ├── supervisor/        # Monthly PhD briefing reports
│   └── PHD_ROADMAP.md     # 3-year PhD thesis strategy
├── paper/                 # Live working manuscript (paper/main.md)
├── paper_figures/         # Vector PDF and PNG publication plots (Figures 1–4)
├── results/               # Statistical analysis, calibration, and trace audit JSON reports
├── scripts/
│   ├── analysis/          # Statistical testing, calibration (ECE/Brier/AURC), and plot generation
│   ├── hpc/               # Autonomous 24/7 SLURM campaign daemons & submission scripts
│   └── macbook/           # Bidirectional sync and backup scripts
├── slurm/                 # Reproducible SLURM submission templates
├── AGENTS.md              # Master operating system & AI agent memory
├── CHANGELOG.md           # Chronological execution & code modification log
└── TODO_LIST.md           # Granular experiment roadmap and milestone tracker
```

---

## 4. Hardware & SLURM Execution Guidelines

All experiments are conducted on the **PARAM Rudra HPC** (IIT Patna) under strict resource controls:
* **Partition:** `gpu` partition with NVIDIA A100-PCIE-80GB GPUs.
* **Allocation Policy:** Exactly 1 GPU per job (`--gres=gpu:1`), 2 GPUs max concurrently per user (`QOSMaxGRESPerUser`).
* **Serving Stack:** Pinned `qrm-official` environment (`vLLM==0.7.0`, `--enforce-eager`, `--gpu-memory-utilization 0.75`).
* **AWQ Requirement:** Always pass `--dtype float16` when serving AWQ checkpoints.

---

## 5. Autonomous 24/7 Queue Manager Daemon

The campaign is managed autonomously via `scripts/hpc/queue_manager_daemon.py`:
```bash
# Launch background queue daemon in tmux
tmux new-session -d -s gpqa_daemon "python3 scripts/hpc/queue_manager_daemon.py --config configs/campaign_cells_gpqa.json"
```
Features:
* Maintains continuous 2-channel execution (1 GPU Qwen + 1 GPU Llama) without exceeding cluster submit limits.
* Real-time Telegram progress alerts and hourly rollup dashboards.
* Automatic completion detection and self-healing retries for preempted jobs.

---

## 6. Citation & Reproducibility Package

All raw per-problem outputs, evaluation manifests, and validation JSONs are preserved in `archive/outputs-hpc-campaign-2026-08-14/` and `archive/outputs-hpc-breadth-gsm8k-2026-08-15/`.

```bibtex
@article{nandish2026beyondpass1,
  title={Beyond Pass@1: Reliability--Cost Frontiers of Quantized Reasoning Models under Controlled Serving-Stack Shift},
  author={Nandish, Manish},
  journal={Working Draft},
  institution={Indian Institute of Technology Patna},
  year={2026}
}
```
