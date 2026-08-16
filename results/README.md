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

Cost-of-Pass in the paper is a **fixed-throughput token-cost proxy** ($\$1.50$/A100-h, assumed $65$ tok/s). Not measured tokens/sec or VRAM.

---

## 4. What is not in these JSON files

Compact validation records have `extractive_match`, `completion_tokens`, and repetition flags. They do **not** contain extracted answer strings, token IDs, raw text, or vLLM `finish_reason`.

Therefore:

* Modal-answer selective prediction is **not yet available**.
* Guo-style ECE/Brier from model probabilities is not available.
* Gold-hit $k/5$ histograms are oracle diagnostics, not a deployable abstention gate.

If HPC JSONLs still exist, export answers with `scripts/hpc/qrm_parity/check_campaign_jsonls.sh` using the frozen policy in [`docs/ANSWER_NORMALIZATION.md`](../docs/ANSWER_NORMALIZATION.md).

---

## 5. Directory structure

```
results/
├── math500/      # 40 official validation records
├── gsm8k/        # 24 breadth records
├── gpqa/         # 24 breadth records
├── reports/
│   ├── revision_reanalysis_report.json   # CANONICAL
│   └── runtime_manifest.json             # effective 56k launch settings
└── README.md
```

Other files under `reports/` are either derived stubs or historical. Cite only `revision_reanalysis_report.json`.
