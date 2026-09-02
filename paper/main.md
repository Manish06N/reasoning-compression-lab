# One Stack, Many Rankings: Measuring Evaluation-Target Instability in Quantized Reasoning Checkpoints

**Manish Nandish**<sup>1,2</sup>, **Rajiv Misra**<sup>1</sup>, and **Midhunchakkaravarthy Janarthanan**<sup>2</sup>

<sup>1</sup> Department of Computer Science and Engineering, Indian Institute of Technology Patna, Patna, Bihar, India
Email: {manish_25s21res58, rajivm}@iitp.ac.in

<sup>2</sup> Lincoln University College, Malaysia
Email: Midhunchakkaravarthy@lincoln.edu.my

**Do not cite this markdown file for numbers.** The canonical manuscript is [`main.tex`](main.tex) compiled to [`main.pdf`](main.pdf). This file exists only so repository markdown matches the LaTeX story.

**Keywords:** Reasoning language models, public quantization checkpoints, pinned serving stack, estimand disagreement, Cost-of-Pass.

---

## Abstract (same claims as `main.tex`)

Practitioners often select a public quantized reasoning checkpoint using one published metric, typically pass@1. Prior measurement studies examine individual dimensions—accuracy, serving throughput, token length, or calibration—but do not ask whether checkpoint rankings remain stable when the serving stack is frozen and the evaluation target changes.

We pin one stack and vary only the public weight checkpoint. Eight DeepSeek-R1-Distill checkpoints (BF16; FP8 executed as Marlin W8A16 rather than native W8A8; AWQ-4; GPTQ-4; Qwen-7B and Llama-8B) are evaluated on an NVIDIA A100-80GB with vLLM 0.7.0 eager execution. The campaign comprises 88 checkpoint×benchmark×seed runs and 56,408 completions on MATH-500 (5 seeds), GSM8K (3 seeds), and GPQA-Diamond (3 seeds).

Under this pin, rankings disagree across estimands. On MATH-500, FP8–BF16 pass@1 differences are $+0.40$ and $+0.28$ percentage points (pp); problem-clustered 95% intervals include zero, and a $\pm 1$ pp equivalence test is not passed. The tested community AWQ artifacts showed task-specific degradation. The tested Qwen AWQ artifact exhibited a $5.56$ pp GPQA-Diamond difference under the primary Holm-6 family; this contrast does not remain significant under the Holm-18 joint sensitivity analysis. The tested Qwen 4-bit checkpoints showed $6.3$–$6.9\%$ higher mean MATH-500 completion length, including among jointly correct pairs. Historical token-implied cost, sequential GPU-seconds (Condition A), and batched GPU-seconds (Condition B) rank the tested Qwen cells differently. The deployment ranking of a quantized reasoning checkpoint depends on the checkpoint, task, estimand, and serving condition under the evaluated stack.

---

## Research questions

1. **RQ1.** Do the evaluated quantized checkpoints differ in *pass@1* from matched BF16, with problem-clustered uncertainty?
2. **RQ2.** How do completion length and correctness-conditioned length differ across the evaluated checkpoints, and what do identical-word loops and near-cap completions reveal about the long-tail behavior?
3. **RQ3.** What can observable multi-sample agreement say about selective abstention without gold labels at serve time?
4. **RQ4.** Do checkpoint rankings agree across the historical token proxy, sequential Condition A, and batched Condition B aggregate serving-cost proxies?

This paper **pins** one stack. It does not run a factorial vLLM 0.7.0 vs 0.8.5 experiment. Contributions in `main.tex` are (C1) pinned evaluation protocol, (C2) ranking instability, (C3) checkpoint-not-method (tested community AWQ artifacts showed task-specific degradation).

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
| Cost | Aggregate hybrid Cost-of-Pass proxy $\widetilde{C}_{\mathrm{pass}}^{\mathrm{hyb}}$: confirmation GPU-seconds / campaign MATH pass@1. Rankings disagree across 65 tok/s proxy, Condition A, and Condition B (serving-condition sensitivity, not isolated batching). Qwen FP8 B: five-rep listing, not a lone $-36.0\%$. |
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
