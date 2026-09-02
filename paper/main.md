# One Stack, Many Rankings: Evaluating Quantized Reasoning Checkpoints Beyond Accuracy

**Manish Nandish**<sup>1,2</sup>, **Rajiv Misra**<sup>1</sup>, and **Midhunchakkaravarthy Janarthanan**<sup>2</sup>

<sup>1</sup> Department of Computer Science and Engineering, Indian Institute of Technology Patna, Patna, Bihar, India
Email: {manish_25s21res58, rajivm}@iitp.ac.in

<sup>2</sup> Lincoln University College, Malaysia
Email: Midhunchakkaravarthy@lincoln.edu.my

**Do not cite this markdown file for numbers.** The canonical manuscript is [`main.tex`](main.tex) compiled to [`main.pdf`](main.pdf). This file exists only so repository markdown matches the LaTeX story.

**Keywords:** Reasoning language models, public quantization checkpoints, pinned serving stack, estimand disagreement, Cost-of-Pass.

---

## Abstract (same claims as `main.tex`)

Public quantized reasoning checkpoints do not receive a single deployment ranking: under one pinned serving stack, rank order depends on the checkpoint, task, evaluation target, estimand, and serving condition. We evaluate public DeepSeek-R1-Distill checkpoints on an NVIDIA A100-80GB with vLLM 0.7.0 eager execution, varying only the public weight checkpoint among BF16, FP8 executed as Marlin W8A16 rather than native W8A8, AWQ-4, and GPTQ-4 for DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B. The evaluation comprises 88 checkpoint×benchmark×seed runs and 56,408 completions on MATH-500 (5 seeds), GSM8K (3 seeds), and GPQA-Diamond (3 seeds).

Observed under this pinned stack, the tested checkpoints change rank across evaluation targets. On MATH-500, FP8–BF16 pass@1 differences are $+0.40$ and $+0.28$ percentage points (pp); problem-clustered 95% intervals include zero, and a $\pm 1$ pp equivalence test is not passed. The tested community Qwen AWQ-4 artifact showed $5.56$ pp lower pass@1 on GPQA-Diamond. The Qwen AWQ GPQA result is significant within the primary Holm-6 family, but not under the Holm-18 joint sensitivity analysis. The tested Qwen 4-bit checkpoints showed $6.3$–$6.9\%$ higher mean MATH-500 completion length. The historical token proxy, sequential GPU-seconds (Condition A), and batched GPU-seconds (Condition B) produce different rankings among the tested Qwen cells. The deployment ranking of a quantized reasoning checkpoint is not a property of bit-width alone.

---

## Research questions

1. **RQ1.** Do the evaluated quantized checkpoints differ in *pass@1* from matched BF16, with problem-clustered uncertainty?
2. **RQ2.** How do completion length and correctness-conditioned length differ across the evaluated checkpoints, and what do identical-word loops and near-cap completions reveal about the long-tail behavior?
3. **RQ3.** What can observable multi-sample agreement say about selective abstention without gold labels at serve time?
4. **RQ4.** Do checkpoint rankings agree across the historical token proxy, sequential Condition A, and batched Condition B aggregate serving-cost proxies?

This paper **pins** one stack. It does not run a factorial vLLM 0.7.0 vs 0.8.5 experiment. Contributions in `main.tex` are (C1) pinned protocol, (C2) ranking instability, (C3) checkpoint-not-method (tested community AWQ artifacts only).

---

## Canonical findings (aligned with `main.tex`)

| Topic | Claim in this manuscript |
|---|---|
| Serving stack | Pinned `qrm-official` / vLLM 0.7.0 eager / A100 W8A16 FP8 fallback |
| Pathology | 25 loop-flagged completions; 0 exact cap hits; 209 near-cap completions ($\ge 32{,}500$ tokens); Qwen AWQ/GPTQ MATH near-cap $25$/$24$ vs BF16 $14$ |
| Llama AWQ-4 | Tested `jakiAJK` checkpoint: significant MATH-500 and GSM8K pass@1 drop vs BF16 |
| Qwen AWQ-4 | Tested community artifact: −5.56 pp GPQA-Diamond pass@1. The Qwen AWQ GPQA result is significant within the primary Holm-6 family, but not under the Holm-18 joint sensitivity analysis |
| Qwen 4-bit tokens | $+6.3$–$6.9\%$ RoM vs BF16; Both-OK CIs exclude 0; mismatch-conditioned $D$ is a diagnostic (not causal); BF16-correct conditional $\Delta$, following Lian et al., positive |
| 200-item subset | Superseded estimator (Appendix); not a result |
| Modal-answer selective prediction | Secondary gold-free unique-mode abstention; 5/5 observed risk $\le 0.27\%$; Wilson upper bounds on $0/n$ cells 0.82%–1.08%. Not G-Pass@k. Not a safety property. |
| Cost | Aggregate hybrid Cost-of-Pass proxy $\widetilde{C}_{\mathrm{pass}}^{\mathrm{hyb}}$: confirmation GPU-seconds / campaign MATH pass@1. Rankings disagree across 65 tok/s proxy, Condition A, and Condition B (serving-condition sensitivity, not isolated batching). Qwen FP8 B: five-rep listing, not a lone $-36.0\%$. |
| FP8 vs BF16 | 95% CIs include 0; TOST $\pm 1$ pp **fails**; not claimed equivalent |

Tables, TikZ figures, limitations, and the appendix live in `main.tex` / `main.pdf`. Frozen analysis tables: `results/reports/major_revision_tables.md`. Reproduce numbers with:

```bash
python3 scripts/analysis/revision_reanalysis.py --check
python3 scripts/hpc/qrm_parity/benchmark_serving_confirmation.py --check
python3 scripts/analysis/emit_major_revision_tables.py --check
```

See [`../REPRODUCE.md`](../REPRODUCE.md).

## Artifacts

https://github.com/Manish06N/reasoning-compression-lab

Per-cell records: `results/math500/`, `results/gsm8k/`, `results/gpqa/`. Canonical report: `results/reports/revision_reanalysis_report.json`. Frozen tables: `results/reports/major_revision_tables.md`. Modal agreement: `results/reports/modal_agreement_report.json`. ArXiv zip: [`arxiv_source.zip`](arxiv_source.zip). Packaging: [`ARTIFACT.md`](ARTIFACT.md).
