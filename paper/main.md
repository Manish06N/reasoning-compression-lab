# Pitfalls of Single-Metric Evaluation for Quantized Reasoning Checkpoints

**Manish Nandish**<sup>1,2</sup>, **Rajiv Misra**<sup>1</sup>, and **Midhunchakkaravarthy Janarthanan**<sup>2</sup>

<sup>1</sup> Department of Computer Science and Engineering, Indian Institute of Technology Patna, Patna, Bihar, India
Email: {manish_25s21res58, rajivm}@iitp.ac.in

<sup>2</sup> Lincoln University College, Malaysia
Email: midhun@lincoln.edu.my

**Do not cite this markdown file for numbers.** The canonical manuscript is [`main.tex`](main.tex) compiled to [`main.pdf`](main.pdf). This file exists only so repository markdown matches the LaTeX story.

**Keywords:** Reasoning language models, public quantization checkpoints, pinned serving stack, estimand disagreement, Cost-of-Pass.

---

## Abstract (same claims as `main.tex`)

Practitioners often select a public quantized reasoning checkpoint from a single metric. We pin one A100 stack and evaluate eight DeepSeek-R1-Distill checkpoints (88 runs, 56,408 completions).

The robust accuracy drop is Llama AWQ-4 (MATH-500 −2.76 pp; GSM8K −1.57 pp). maj@5 cuts that MATH gap to about −1.4 pp. A BF16-correct length contrast is positive in every cell; the quant-correct mirror flips sign in four of six, and three of those intervals exclude zero. Subset serving cost is one length draw: Llama FP8 is about 7,288 tokens on Condition A and 4,551 on the full grid. Five-sample agreement has selective risk at most 0.27%, against 1.6–4.6% for a one-sample length rule at matched coverage.

---

## Research questions

1. **RQ1.** Do the evaluated quantized checkpoints differ in *pass@1* from matched BF16, with problem-clustered uncertainty?
2. **RQ2.** How do completion length and correctness-conditioned length differ across the evaluated checkpoints, and what do identical-word loops and near-cap completions reveal about the long-tail behavior?
3. **RQ3.** What can observable multi-sample agreement say about selective abstention without gold labels at serve time?
4. **RQ4.** Do checkpoint rankings agree across the historical token proxy, sequential Condition A, and batched Condition B aggregate serving-cost proxies?

This paper **pins** one stack. Contributions in `main.tex` are (C1) pinned protocol with recorded dtype and kernel, (C2) estimator-sensitive point orders, with length variance stated, (C3) AWQ results scoped to the jakiAJK artifacts.

**Novelty defense (same claim as related work in `main.tex`):** Existing studies evaluate quantization accuracy, throughput, or individual reasoning behaviors. Our question is different: after fixing the serving stack, do practitioners receive the same checkpoint recommendation when the evaluation target changes? We study ranking stability rather than proposing another quantization method.

**Venue:** Journal of Systems and Software (JSS) first. This 22-page single-column PDF is the initial-submission form (Elsevier Your Paper Your Way). Do not send it unchanged to TMLR or FGCS. See [`../docs/VENUE.md`](../docs/VENUE.md).

---

## Canonical findings (aligned with `main.tex`)

| Topic | Claim in this manuscript |
|---|---|
| Serving stack | Pinned `qrm-official` / vLLM 0.7.0 eager / A100 W8A16 FP8 fallback |
| Pathology | 25 loop-flagged completions; 0 exact cap hits; 209 near-cap completions ($\ge 32{,}500$ tokens); tested Qwen AWQ/GPTQ MATH near-cap $25$/$24$ vs BF16 $14$ |
| Llama AWQ-4 | Tested `jakiAJK` checkpoint: significant MATH-500 and GSM8K pass@1 drop vs BF16 |
| Qwen AWQ-4 | Not a headline. Tested community artifact: 5.56 pp GPQA-Diamond difference under Holm-6; not significant under Holm-18 joint sensitivity. 75 of 198 items flip on at least one seed; 0 all-three-seed flips |
| Qwen 4-bit tokens | $+6.3$–$6.9\%$ RoM vs BF16; Both-OK CIs exclude 0; mismatch-conditioned $D$ is a diagnostic (not causal); BF16-correct conditional $\Delta$, following Lian et al., positive |
| 200-item subset | Superseded estimator (Appendix); not a result |
| Modal-answer selective prediction | Secondary gold-free unique-mode abstention; 5/5 observed risk $\le 0.27\%$; Wilson upper bounds on $0/n$ cells 0.82%–1.08%. Not G-Pass@k. Not a safety property. |
| Cost | Subset GPU-seconds and a campaign-length sensitivity (tokens / measured tok/s / pass@1) do not share a point order. Timing intervals are wall-clock repeats of one seed. Qwen FP8 B tok/s $449.79$ is a mean of ratios; $8.64$ GPU-s/q implies about $432.5$ tok/s. |
| FP8 vs BF16 | 95% CIs include 0; TOST $\pm 1$ pp **fails**; not claimed equivalent |

Tables, TikZ figures, limitations, and the appendix live in `main.tex` / `main.pdf`. Frozen analysis tables: `results/reports/major_revision_tables.md`. Reproduce numbers with:

```bash
python3 scripts/analysis/revision_reanalysis.py --check
python3 scripts/hpc/qrm_parity/benchmark_serving_confirmation.py --check
python3 scripts/analysis/emit_major_revision_tables.py --check
python3 scripts/check_tex_tables.py --check
```

See [`../REPRODUCE.md`](../REPRODUCE.md).

## Artifacts

https://github.com/Manish06N/reasoning-compression-lab

Per-cell records: `results/math500/`, `results/gsm8k/`, `results/gpqa/`. Canonical report: `results/reports/revision_reanalysis_report.json`. Frozen tables: `results/reports/major_revision_tables.md`. Modal agreement: `results/reports/modal_agreement_report.json`. ArXiv zip: [`arxiv_source.zip`](arxiv_source.zip). Packaging: [`ARTIFACT.md`](ARTIFACT.md).

## CRediT (same as `main.tex`)

- **Manish Nandish:** Conceptualization, Methodology, Software, Investigation, Data curation, Formal analysis, Visualization, Writing – original draft, Writing – review & editing.
- **Rajiv Misra:** Conceptualization, Resources, Supervision, Writing – review & editing.
- **Midhunchakkaravarthy Janarthanan:** Supervision, Writing – review & editing.

Dual affiliation of the first author is a joint IIT Patna–Lincoln doctoral arrangement. No experimental or software roles are claimed for Midhunchakkaravarthy Janarthanan.
