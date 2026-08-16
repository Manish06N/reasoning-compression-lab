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
Canonical preprint: [`paper/main.tex`](paper/main.tex) → [`paper/main.pdf`](paper/main.pdf) (12 pages). Scoreboard: [`results/README.md`](results/README.md). **88 cells / 56,408 completions** (MATH-500 40 + GSM8K 24 + GPQA-Diamond 24).

All 40 MATH-500 cells under the pinned `qrm-official` stack (`vLLM 0.7.0` eager, $T=0.6, p=0.95, \text{max\_tokens}=32,768$) show **0 heuristic truncations** and **0 identical-word loops** (token-cap and consecutive-word detectors; official QRM rows do not store `finish_reason`):

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

### Breadth Benchmark Results (sample std over seeds)
* **GSM8K** ($n=1{,}319$, 3 seeds): Qwen BF16 $91.26\% \pm 0.29\%$; FP8 $91.33\% \pm 0.16\%$; AWQ-4 $91.05\% \pm 1.14\%$; GPTQ-4 $91.13\% \pm 0.27\%$. Llama BF16 $88.68\% \pm 0.46\%$; FP8 $88.80\% \pm 0.62\%$; AWQ-4 $87.11\% \pm 0.23\%$; GPTQ-4 $88.96\% \pm 0.70\%$.
* **GPQA-Diamond** ($n=198$, 3 seeds): Qwen BF16 $50.34\% \pm 2.96\%$; FP8 $49.49\% \pm 1.52\%$; AWQ-4 $44.78\% \pm 3.04\%$; GPTQ-4 $47.98\% \pm 1.75\%$. Llama BF16 $46.13\% \pm 1.91\%$; FP8 $47.81\% \pm 0.29\%$; AWQ-4 $46.97\% \pm 2.02\%$; GPTQ-4 $44.95\% \pm 4.32\%$.

---

## 2. Key Scientific Findings

1. **FP8 vs BF16 (maj@5 McNemar):** On A100 Marlin W8A16 (not native W8A8), paired McNemar tests on *maj@5* find no significant discordance after Holm–Bonferroni ($p > 0.29$). That is not a test of pass@1 means. Llama AWQ-4 still drops $2.76$ pp on mean pass@1.
2. **Architecture sensitivity:** Qwen-7B 4-bit stays above $93.1\%$ on MATH-500; Llama-8B AWQ-4 is the weak cell ($86.48\%$ vs $89.24\%$ BF16).
3. **Pathology heuristics:** Token-cap and identical-word detectors flag 0/0 across all 88 cells. Phrase-level loops and `finish_reason=length` are not in the saved schema.
4. **Token inflation & modeled $C_{\text{pass}}$:** Full-grid MATH-500 4-bit inflation is $+1.7\%$ to $+6.8\%$ vs BF16; the 200-item audit subset is $+10\%$ to $+30\%$. Cost-of-Pass is modeled at $\$1.50$/A100-h and $65$ tok/s (not measured wall-clock). Under that model, FP8 is lowest $C_{\mathrm{pass}}$.

---

## 3. Repository Structure

```
reasoning-compression-lab/
├── configs/               # Campaign cell configurations (MATH-500, GSM8K, GPQA-Diamond)
├── docs/                  # PhD Roadmap, literature surveys, supervisor briefing, hardware policies
│   ├── literature/        # Reading maps and base paper summaries
│   ├── supervisor/        # Monthly PhD briefing reports
│   └── PHD_ROADMAP.md     # 3-year PhD thesis strategy
├── paper/                 # Canonical LaTeX (`main.tex` → `main.pdf`); markdown mirror `main.md`
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
