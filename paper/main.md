# Beyond Pass@1: Accuracy, Agreement, and Serving-Cost Effects of Public R1-Distill Quantization Checkpoints under a Pinned Stack

**Manish Nandish**
Department of Computer Science & Engineering
Indian Institute of Technology Patna, India
Email: manishn_iitp@iitp.ac.in

**Do not cite this markdown file for numbers.** The canonical manuscript is [`main.tex`](main.tex) compiled to [`main.pdf`](main.pdf). This file exists only so repository markdown matches the LaTeX story.

**Keywords:** Reasoning language models, public quantization checkpoints, pinned serving stack, estimand disagreement, Cost-of-Pass.

---

## Abstract (same claims as `main.tex`)

Post-training quantization is a common serving default for reasoning language models, but a single pass@1 or token count is not a complete evaluation. We evaluate public R1-Distill checkpoints under one pinned serving stack—vLLM 0.7.0 eager execution on an NVIDIA A100-80GB, with FP8 checkpoints executed as Marlin W8A16 rather than native W8A8—and vary only the public weight checkpoint among BF16, FP8, AWQ-4, and GPTQ-4 for DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B. The evaluation comprises 88 checkpoint×benchmark×seed runs and 56,408 completions on MATH-500 (5 seeds), GSM8K (3 seeds), and GPQA-Diamond (3 seeds).

Under this pin, estimands disagree. On MATH-500, FP8–BF16 pass@1 differences are $+0.40$ and $+0.28$ percentage points (pp); problem-clustered 95% intervals include zero, and a $\pm 1$ pp equivalence test is not passed. The tested Qwen AWQ-4 checkpoint loses $5.56$ pp on GPQA-Diamond (Holm-significant within the primary six GPQA contrasts, not under a joint 18-contrast Holm sensitivity). Mean MATH-500 length rises $6.3$–$6.9\%$ for the tested Qwen 4-bit checkpoints, including detectable lengthening among jointly correct Qwen AWQ-4/GPTQ-4 pairs; much larger conditional differences occur in mismatch cases, where failure traces are themselves long. Strict five-sample unique-mode agreement yields low observed selective risk, trading off coverage and requiring five generations. The historical token proxy, sequential Condition A, and batched Condition B produce different rankings: the proxy ranks Qwen FP8 first among Qwen cells, Condition A ranks Qwen AWQ-4 first, and Condition B ranks Qwen GPTQ-4 first. Effects and rankings are checkpoint-, task-, workload-, estimand-, and serving-condition-specific under this pinned stack.

---

## Research questions

1. **RQ1.** Does quantization change *pass@1* relative to matched BF16, with problem-clustered uncertainty?
2. **RQ2.** How do completion length and correctness-conditioned length differ across the evaluated checkpoints, and what do identical-word loops and near-cap completions reveal about the long-tail behavior?
3. **RQ3.** What can be said about multi-sample reliability without gold labels at serve time?
4. **RQ4.** Do checkpoint rankings agree across the historical token proxy, sequential Condition A, and batched Condition B aggregate serving-cost proxies?

This paper **pins** one stack. It does not run a factorial vLLM 0.7.0 vs 0.8.5 experiment.

---

## Canonical findings (aligned with `main.tex`)

| Topic | Claim in this manuscript |
|---|---|
| Serving stack | Pinned `qrm-official` / vLLM 0.7.0 eager / A100 W8A16 FP8 fallback |
| Pathology | 25 loop-flagged completions; 0 exact cap hits; 209 near-cap completions ($\ge 32{,}500$ tokens); Qwen AWQ/GPTQ MATH near-cap $25$/$24$ vs BF16 $14$ |
| Llama AWQ-4 | Tested `jakiAJK` checkpoint: significant MATH-500 and GSM8K pass@1 drop vs BF16 |
| Qwen AWQ-4 | Tested checkpoint: −5.56 pp GPQA-Diamond pass@1; Holm-significant within the primary six GPQA contrasts, not under Holm-18 |
| Qwen 4-bit tokens | $+6.3$–$6.9\%$ RoM vs BF16; Both-OK CIs exclude 0; mismatch-conditioned $D$ is a diagnostic (not causal); BF16-correct conditional $\Delta$, following Lian et al., positive |
| 200-item subset | Superseded estimator (Appendix); not a result |
| Modal-answer selective prediction | Secondary gold-free unique-mode abstention; 5/5 observed risk $\le 0.27\%$; Wilson upper bounds on $0/n$ cells $0.79$–$1.08\%$. Not G-Pass@k. Not a safety property. |
| Cost | Aggregate hybrid Cost-of-Pass proxy $\widetilde{C}_{\mathrm{pass}}^{\mathrm{hyb}}$: confirmation GPU-sec / campaign MATH pass@1. Rankings disagree across 65 tok/s proxy, Condition A, and Condition B (serving-condition sensitivity, not isolated batching). Qwen FP8 B: five-rep listing, not a lone $-36.0\%$. |
| FP8 vs BF16 | 95% CIs include 0; TOST $\pm 1$ pp **fails**; not claimed equivalent |

Tables, TikZ figures, limitations, and the appendix live in `main.tex` / `main.pdf`. Frozen analysis tables: `results/reports/major_revision_tables.md`. Reproduce numbers with:

```bash
python3 scripts/analysis/revision_reanalysis.py --check
python3 scripts/analysis/modal_agreement_analysis.py --check
python3 scripts/analysis/measured_serving_analysis.py --check
python3 scripts/analysis/measured_serving_confirmation_analysis.py --check
python3 scripts/analysis/emit_major_revision_tables.py
python3 scripts/hpc/qrm_parity/validate_measured_serving_confirmation.py
```

## Artifacts

https://github.com/Manish06N/reasoning-compression-lab

Per-cell records: `results/math500/`, `results/gsm8k/`, `results/gpqa/`. Canonical report: `results/reports/revision_reanalysis_report.json`. Frozen tables: `results/reports/major_revision_tables.md`. Modal agreement: `results/reports/modal_agreement_report.json`. ArXiv zip: [`arxiv_source.zip`](arxiv_source.zip). Packaging: [`ARTIFACT.md`](ARTIFACT.md).
