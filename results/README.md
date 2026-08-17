# Experimental Results & Validation Archive

**Repository:** reasoning-compression-lab
**Cluster:** PARAM Rudra HPC (NVIDIA A100-PCIE-80GB)
**Canonical numbers:** [`reports/revision_reanalysis_report.json`](reports/revision_reanalysis_report.json)
**Recompute:** `python3 scripts/analysis/revision_reanalysis.py`
**Drift check:** `python3 scripts/analysis/revision_reanalysis.py --check`

This directory holds the released per-cell records for the **pinned vLLM 0.7.0 eager** campaign (88 cells, 56,408 completions). Do not cite older “0 truncations / 0 loops”, “98.23% operational safety gate”, “FP8 matches BF16”, “FP8 Pareto-optimal”, or “200-item stratified/mixed-correctness” claims. Those are retracted.

---

## 1. Benchmark matrices (pass@1)

### MATH-500 ($n=500$, 5 seeds)

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

Clustered bootstrap vs BF16 (primary): Llama AWQ-4 **−2.76 pp** (95% CI $[−4.16,−1.44]$, $p<0.001$). FP8 CIs include 0; MATH $\pm 1$ pp TOST is not passed.

### GSM8K ($n=1{,}319$, 3 seeds)

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

Llama AWQ-4 vs BF16: **−1.57 pp** ($p=0.0018$).

### GPQA-Diamond ($n=198$, 3 seeds)

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

Qwen AWQ-4 vs BF16: **−5.56 pp** (95% CI $[−9.60,−1.52]$, $p=0.0068$).

---

## 2. Pathology (full grid)

Detectors (see `scripts/analysis/revision_reanalysis.py`):

* **Loop:** longest consecutive identical-word run $\ge 20$. JSON key: `repetition_rows` / per-row `repetition_flag`.
* **Exact cap:** encoded completion length $\ge 32{,}768$. JSON key: `token_limit_hits`.
* **Near-cap proxy:** `completion_tokens >= 32{,}500` (heuristic; `finish_reason` not stored).

| Totals over 56,408 completions | Count |
|---|---|
| Identical-word loops | **25** |
| Exact cap hits | **0** |
| Near-cap ($\ge 32{,}500$ tokens) | **209** |

MATH-500 Qwen 4-bit near-cap counts are about $1.7\times$ the matched BF16 cell. Phrase-level / n-gram cycles and `P(finish_reason=length | tokens >= 32500)` are **not** in this release.

---

## 3. Tokens and cost

Full MATH-500 grid, all 5 seeds, **ratio of means** vs BF16:

| Contrast | Ratio of means |
|---|---|
| Qwen FP8 | −0.09% |
| Qwen AWQ-4 | **+6.33%** |
| Qwen GPTQ-4 | **+6.88%** |
| Llama FP8 | −2.27% |
| Llama AWQ-4 | +1.72% |
| Llama GPTQ-4 | +3.95% |

Cost-of-Pass in the paper has two layers: (1) **primary** measured GPU-sec/query on the frozen 100-problem MATH-500 serving subset (`results/measured_serving/`, report `reports/measured_serving/measured_serving_report.json`) at a $\$1.50$/A100-h scenario; (2) **historical** shared-$65$ tok/s token proxy. Do not cite a unique “true Pareto optimum.”

Batched Condition B (100-prompt `llm.generate`, $R=3$ wall-clock repeats; campaign MATH-500 pass@1):

| Cell | tok/s | GPU-s/q | VRAM (GB) | $C_{\mathrm{pass}}$ |
|---|---|---|---|---|
| Qwen BF16 | $694.27\pm 0.65$ | 5.62 | 56.00 | \$0.0025 |
| Qwen FP8 | $824.27\pm 4.00$ | 4.52 | 56.02 | \$0.0020 |
| Qwen AWQ-4 | $687.51\pm 0.59$ | 5.89 | 56.00 | \$0.0026 |
| Qwen GPTQ-4 | $602.65\pm 1.40$ | 6.70 | 55.17 | \$0.0030 |
| Llama BF16 | $649.72\pm 0.51$ | 6.86 | 56.71 | \$0.0032 |
| Llama FP8 | $759.48\pm 6.96$ | 6.37 | 56.72 | \$0.0030 |
| Llama AWQ-4 | $545.68\pm 125.62$ | 10.15 | 56.72 | \$0.0049 |
| Llama GPTQ-4 | $785.55\pm 12.33$ | 6.58 | 56.74 | \$0.0031 |

Qwen FP8 vs BF16 batched: **+18.7%** tok/s, **−19.8%** $C_{\mathrm{pass}}$. Llama AWQ-4 std includes a mixed-node slower repeat. Peak VRAM is the 0.75 utilization pool.

Reproduce:

```bash
python3 scripts/analysis/measured_serving_analysis.py --check
```

---

## 4. Modal agreement (recovered MATH-500 answers)

Compact validation records still omit traces and `finish_reason`. Recovered extracted answers live in [`recovered/math500_modal_inputs.jsonl`](recovered/math500_modal_inputs.jsonl) (20,000 rows; SHA256 `23e9ead021111959cf047323572889c95be0496e9475d6870b06c8b2c9a6149b`). Canonical modal numbers: [`reports/modal_agreement_report.json`](reports/modal_agreement_report.json).

Gold is used only after unique-mode clustering. Serve at $k/5$ iff a unique modal class has size $\ge k$. Mean five-sample token-cost proxy $T_5$ sums all five seeds before abstention. Campaign clustering used LightEval 0.8.0; MacBook `--check` validates the compact artifact and does not re-extract from `generated_text`.

| Cell | >=3/5 cov / risk | >=4/5 cov / risk | 5/5 cov / risk | Mean $T_5$ |
|---|---|---|---|---|
| Qwen BF16 | 96.0 / 1.67 | 93.4 / 0.43 | 88.4 / 0.23 | 20,057 |
| Qwen FP8 | 96.6 / 1.66 | 92.6 / 0.00 | 88.8 / 0.00 | 20,038 |
| Qwen AWQ-4 | 95.8 / 1.46 | 91.2 / 0.44 | 86.4 / 0.23 | 21,327 |
| Qwen GPTQ-4 | 95.4 / 1.47 | 92.8 / 0.43 | 86.8 / 0.00 | 21,436 |
| Llama BF16 | 94.0 / 2.98 | 87.8 / 0.68 | 76.2 / 0.26 | 23,283 |
| Llama FP8 | 94.0 / 3.19 | 88.6 / 0.90 | 78.2 / 0.00 | 22,754 |
| Llama AWQ-4 | 93.2 / 3.65 | 84.6 / 1.89 | 70.2 / 0.00 | 23,682 |
| Llama GPTQ-4 | 93.8 / 2.77 | 86.4 / 0.46 | 75.4 / 0.27 | 24,203 |

Coverage/risk in percent. A 0.00 risk point estimate is not proof of zero true error. Llama AWQ-4 5/5 coverage vs BF16: **−6.0 pp** (95% paired CI $[-9.4,-2.6]$). Qwen AWQ-4 $\ge 4/5$: **−2.2 pp** ($[-4.0,-0.6]$). FP8 coverage CIs include 0; that is not equivalence.

Gold-hit $k/5$ histograms remain oracle diagnostics, not a deployable abstention gate. Guo-style ECE from model probabilities is still unavailable.

Reproduce:

```bash
python3 scripts/analysis/modal_agreement_analysis.py --check
```

---

## 5. Directory structure

```
results/
├── math500/      # 40 official validation records
├── gsm8k/        # 24 breadth records
├── gpqa/         # 24 breadth records
├── reports/
│   ├── revision_reanalysis_report.json   # CANONICAL pass@1 / pathology / tokens
│   ├── modal_agreement_report.json       # gold-free MATH-500 modal agreement
│   ├── measured_serving/                 # tok/s, latency, VRAM, measured C_pass
│   └── runtime_manifest.json             # effective 56k launch settings
├── measured_serving/
│   ├── input_subset.json                 # frozen 100 MATH-500 prompts (use full_prompt)
│   └── raw/                              # 48 task-realistic + 8 microbenchmark JSON
├── recovered/
│   └── math500_modal_inputs.jsonl        # compact extracted answers (no CoT)
└── README.md
```

Other files under `reports/` are either derived stubs or historical. Cite `revision_reanalysis_report.json` for pass@1/pathology/tokens and `modal_agreement_report.json` for observable agreement.
