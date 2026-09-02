# Reasoning Compression Lab (`reasoning-compression-lab`)

Evaluation harness and artifacts for **Paper 1**: quantized reasoning models under **one pinned serving stack**.

* **Authors:** Manish Nandish (IIT Patna; Lincoln University College), Rajiv Misra (IIT Patna), Midhunchakkaravarthy Janarthanan (Lincoln University College)
* **Cluster:** PARAM Rudra HPC (C-DAC / NSM), NVIDIA A100-PCIE-80GB
* **GitHub:** [https://github.com/Manish06N/reasoning-compression-lab](https://github.com/Manish06N/reasoning-compression-lab)
* **Paper 1 (J1):** *One Stack, Many Rankings: Evaluating Quantized Reasoning Checkpoints Beyond Accuracy*

Canonical manuscript: [`paper/main.tex`](paper/main.tex) → [`paper/main.pdf`](paper/main.pdf). Scoreboard: [`results/README.md`](results/README.md). Canonical numbers: [`results/reports/revision_reanalysis_report.json`](results/reports/revision_reanalysis_report.json). Frozen tables: [`results/reports/major_revision_tables.md`](results/reports/major_revision_tables.md). Modal agreement: [`results/reports/modal_agreement_report.json`](results/reports/modal_agreement_report.json). Serving confirmation: [`results/reports/measured_serving_confirmation/`](results/reports/measured_serving_confirmation/). ArXiv source: [`paper/arxiv_source.zip`](paper/arxiv_source.zip). Packaging: [`paper/ARTIFACT.md`](paper/ARTIFACT.md). Canonical branch: `paper-major-revision` (science frozen at `d707e44`). Immutable snapshot tag after this release: `paper-v1.0-submission`.

**Superseded analyses are provenance only** (200-item length subset, unconstrained serving timing, vLLM 0.8.5 pathology autopsy, gold-hit $k/5$ “safety gate”). Do not mix them with the canonical 56,408-completion reports.

---

## 1. What this repository claims (August 2026)

The published campaign is a **stack-pinned measurement study**: **88 checkpoint×benchmark×seed runs / 56,408 completions** on DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B in BF16, FP8, AWQ-4, and GPTQ-4. Rank order depends on checkpoint, task, evaluation target, estimand, and serving condition. The deployment ranking of a quantized reasoning checkpoint is **not** a property of bit-width alone.

| Benchmark | *n* | Seeds | Completions |
|---|---|---|---|
| MATH-500 | 500 | 42–46 | 20,000 |
| GSM8K | 1,319 | 42–44 | 31,656 |
| GPQA-Diamond | 198 | 42–44 | 4,752 |

**Pinned stack (not a vLLM 0.7 vs 0.8.5 factorial):** `qrm-official`, vLLM **0.7.0** eager, A100-80GB, $T=0.6$, top-$p=0.95$, repetition penalty $1.0$, max new tokens $32{,}768$, `gpu_memory_utilization=0.75`. FP8 checkpoints run as Marlin **W8A16** on A100, not native W8A8. Environment lock: [`requirements-qrm-paper-vllm070.lock`](requirements-qrm-paper-vllm070.lock). Effective launch settings: [`results/reports/runtime_manifest.json`](results/reports/runtime_manifest.json).

Files under `configs/models/` are **not** the campaign launcher (directory holds a warning README only). Historical harness JSON is in [`configs/legacy_models/`](configs/legacy_models/). Those files contain different `max_model_len` / KV-cache defaults and must not be read as the effective 56k stack.

### MATH-500 pass@1 (seed-wise; clustered tests vs BF16 in the paper)

| Model & Format | 42 | 43 | 44 | 45 | 46 | Mean ± Std |
|---|---|---|---|---|---|---|
| Qwen-7B BF16 | 94.4% | 94.0% | 93.8% | 94.6% | 93.2% | **94.00% ± 0.55%** |
| Qwen-7B FP8 | 94.4% | 95.2% | 94.8% | 92.6% | 95.0% | **94.40% ± 1.05%** |
| Qwen-7B AWQ-4 | 92.4% | 92.8% | 93.2% | 93.0% | 94.2% | **93.12% ± 0.67%** |
| Qwen-7B GPTQ-4 | 93.8% | 92.6% | 93.4% | 94.6% | 93.0% | **93.48% ± 0.77%** |
| Llama-8B BF16 | 89.0% | 88.4% | 90.2% | 89.8% | 88.8% | **89.24% ± 0.74%** |
| Llama-8B FP8 | 89.0% | 89.6% | 88.6% | 89.2% | 91.2% | **89.52% ± 1.01%** |
| Llama-8B AWQ-4 | 84.4% | 84.8% | 89.2% | 87.4% | 86.6% | **86.48% ± 1.96%** |
| Llama-8B GPTQ-4 | 88.0% | 89.6% | 86.8% | 89.4% | 90.8% | **88.92% ± 1.55%** |

### Canonical findings (use these; ignore older 0/0 / safety-gate text)

1. **Pinned serving stack.** Weight format is the experimental factor. This paper does **not** claim a Serving-Stack Shift result.
2. **Pathology (full 56,408-row grid).** **25** identical-word loop flags (threshold = 20 consecutive identical words). **0** exact $32{,}768$ cap hits after re-encoding. **209** near-cap generations (`completion_tokens >= 32{,}500`). `finish_reason` is not in the compact JSON.
3. **Pass@1 (problem-clustered bootstrap vs BF16).** The tested Llama AWQ-4 artifact: **−2.76 pp** on MATH-500 (95% CI $[−4.16,−1.44]$, $p<0.001$) and **−1.57 pp** on GSM8K. The tested community Qwen AWQ-4 artifact: **−5.56 pp** on GPQA-Diamond (95% CI $[−9.60,−1.52]$, $p=0.007$). The Qwen AWQ GPQA result is significant within the primary Holm-6 family, but not under the Holm-18 joint sensitivity analysis. FP8–BF16 95% intervals include 0; MATH $\pm 1$ pp equivalence is **not** established. maj@5 McNemar is secondary and non-significant.
4. **Completion length (full MATH-500 grid, all 5 seeds, ratio of means).** The tested Qwen AWQ-4 checkpoint showed **+6.33%** and the tested Qwen GPTQ-4 checkpoint **+6.88%** vs matched BF16 (paired token CIs exclude 0). Both-OK clustered CIs exclude 0 for those Qwen 4-bit cells. Mismatch-conditioned length estimates are strongly affected by long failure traces and are **not causal**. The old 200-item even-index mean-of-ratios subset is a superseded estimator.
5. **Selective prediction.** Gold-free unique-mode abstention from recovered MATH-500 answer strings (secondary). Strict 5/5 consensus has observed selective error at most 0.27%; Wilson upper bounds on $0/n$ cells remain strictly positive. The tested Llama AWQ-4 artifact showed **6.0 pp** lower 5/5 coverage vs BF16. Not G-Pass@k. Not a safety property. Gold-hit $k/5$ tables (including the old 98.23% “operational safety gate”) are **not** in the manuscript.
6. **Cost.** Aggregate hybrid Cost-of-Pass proxy $\widetilde{C}_{\mathrm{pass}}^{\mathrm{hyb}}$ uses confirmation GPU-seconds over campaign MATH-500 pass@1. Rankings **disagree** across the historical 65 tok/s token proxy, sequential Condition A, and batched Condition B (serving-condition sensitivity, not isolated batching). Qwen ranking: proxy → FP8 first; Condition A → AWQ-4 first; Condition B → GPTQ-4 first. Qwen GPTQ-4 Condition B is $-45.9\%$ vs BF16 ($95\%$ CI $[-46.4,-45.4]$). Qwen FP8 Condition B is five wall-clock repeats (slow ~351 / mid ~456 / fast ~545 tok/s), not a lone $-36.0\%$; the cause was not identified. Llama GPTQ-4 Condition B mean throughput is about $0.2\%$ lower than Llama BF16. Peak allocated VRAM (~54–56 GB) is the 0.75 engine pool. The 65 tok/s proxy is appendix-only.

### Breadth means (sample std over seeds)

* **GSM8K:** Qwen BF16 $91.26\% \pm 0.29\%$; FP8 $91.33\% \pm 0.16\%$; AWQ-4 $91.05\% \pm 1.14\%$; GPTQ-4 $91.13\% \pm 0.27\%$. Llama BF16 $88.68\% \pm 0.46\%$; FP8 $88.80\% \pm 0.62\%$; AWQ-4 $87.11\% \pm 0.23\%$; GPTQ-4 $88.96\% \pm 0.70\%$.
* **GPQA-Diamond:** Qwen BF16 $50.34\% \pm 2.96\%$; FP8 $49.49\% \pm 1.52\%$; AWQ-4 $44.78\% \pm 3.04\%$; GPTQ-4 $47.98\% \pm 1.75\%$. Llama BF16 $46.13\% \pm 1.91\%$; FP8 $47.81\% \pm 0.29\%$; AWQ-4 $46.97\% \pm 2.02\%$; GPTQ-4 $44.95\% \pm 4.32\%$.

---

## 2. Canonical analysis pipeline

```text
results/{math500,gsm8k,gpqa}/*.json     # released per-cell records
        ↓
scripts/analysis/revision_reanalysis.py # stdlib only; no scratch / HPC JSONL paths
        ↓
results/reports/revision_reanalysis_report.json
        ↓
paper/main.tex tables
```

```bash
python3 scripts/analysis/revision_reanalysis.py --check
python3 scripts/analysis/modal_agreement_analysis.py --check-artifact
python3 scripts/analysis/measured_serving_confirmation_analysis.py --check
python3 scripts/hpc/qrm_parity/benchmark_serving_confirmation.py --check
python3 scripts/analysis/emit_major_revision_tables.py --check
python3 scripts/analysis/item_level_descriptive_analysis.py --check
python3 scripts/hpc/qrm_parity/validate_measured_serving_confirmation.py
```

Older scripts live in [`scripts/analysis/legacy/`](scripts/analysis/legacy/) and must not be used for paper numbers.

Answer-string recovery used the frozen policy in [`docs/ANSWER_NORMALIZATION.md`](docs/ANSWER_NORMALIZATION.md). Compact recovered answers: [`results/recovered/math500_modal_inputs.jsonl`](results/recovered/math500_modal_inputs.jsonl) (SHA256 `23e9ead021111959cf047323572889c95be0496e9475d6870b06c8b2c9a6149b`).

---

## 3. Repository structure

```
reasoning-compression-lab/
├── configs/               # Cell lists; configs/models/ is NOT the 56k launcher
│   ├── models/README.md   # warning only
│   └── legacy_models/     # historical harness JSON (wrong max_model_len / KV defaults)
├── docs/                  # Roadmap, literature, supervisor notes
├── paper/                 # Canonical LaTeX (main.tex → main.pdf); arxiv_source.zip
├── results/               # Per-cell JSON + canonical reports
├── scripts/analysis/      # revision_reanalysis.py + emit_major_revision_tables.py
│   └── legacy/            # Deprecated analysis (wrong keys / gold-hit gate)
├── scripts/hpc/           # SLURM + qrm-official launchers
├── requirements-qrm-paper-vllm070.lock   # 56k campaign environment
├── requirements-hpc-legacy-vllm085.txt   # earlier qreason / vLLM 0.8.5 harness
├── AGENTS.md
└── CHANGELOG.md
```

---

## 4. Hardware & SLURM

* **Partition:** `gpu`, NVIDIA A100-PCIE-80GB, `--gres=gpu:1`. Do not set `#SBATCH --mem`.
* **QOS:** max 2 GPUs concurrent per user.
* **Paper env:** conda env `qrm-official` (not `qreason`). See `scripts/hpc/qrm_parity/install_official_qrm_env.sh`.
* **AWQ:** pass `--dtype float16`.

Do **not** start a new 50k campaign, extra seeds, or a vLLM 0.7 vs 0.8.5 factorial. **Experimental GPU work is closed.** See [`REPRODUCE.md`](REPRODUCE.md) for expected `--check` PASS lines. Tables are reproducible from compact artifacts. Replaying the complete GPU campaign needs original A100 hardware, checkpoint weights, and dataset revisions; full traces are not publicly released.

---

## 5. Citation

```bibtex
@article{nandish2026onestack,
  title={One Stack, Many Rankings: Evaluating Quantized Reasoning Checkpoints Beyond Accuracy},
  author={Nandish, Manish and Misra, Rajiv and Janarthanan, Midhunchakkaravarthy},
  journal={Working Draft},
  year={2026}
}
```
