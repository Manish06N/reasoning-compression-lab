# Pitfalls of Single-Metric Evaluation for Quantized Reasoning Checkpoints

**Manish Nandish**<sup>1,2</sup>, **Rajiv Misra**<sup>1</sup>, and **Midhunchakkaravarthy Janarthanan**<sup>2</sup>

<sup>1</sup> Department of Computer Science and Engineering, Indian Institute of Technology Patna, Patna, Bihar, India
Email: {manish_25s21res58, rajivm}@iitp.ac.in

<sup>2</sup> Lincoln University College, Malaysia
Email: midhun@lincoln.edu.my

**Do not cite this markdown file for numbers.** The canonical manuscript is [`main.tex`](main.tex) compiled to [`main.pdf`](main.pdf). This file exists only so repository markdown matches the LaTeX story.

**Keywords:** Reasoning language models, quantization checkpoints, selection effects, majority vote, selective prediction.

---

## Abstract (same claims as `main.tex`)

Practitioners often choose a public quantized reasoning checkpoint from a single published number. We show three ways that one summary number misleads — a single pass@1, a single correctness-conditioned length estimate, and a single serving subset — using one pinned A100 stack (vLLM 0.7.0, eager) and eight DeepSeek-R1-Distill checkpoints (Qwen-7B and Llama-8B in BF16, FP8 executed as Marlin W8A16, community AWQ-4, and RedHatAI GPTQ-4) on MATH-500, GSM8K, and GPQA-Diamond (88 runs, 56,408 completions).

(i) *Accuracy.* Only one drop survives item-level intervals and multiplicity control: Llama AWQ-4 (MATH-500 −2.76 pp; GSM8K −1.57 pp). Strict-majority voting over five samples (maj@5) halves the MATH gap to −1.4 pp. (ii) *Length.* The BF16-correct token-inflation contrast is biased by selection: comparing two BF16 seeds, with no quantization at all, already yields +138 (Qwen) and +235 (Llama) tokens, and conditioning on the quantized model's success instead flips the sign in four of six contrasts. (iii) *Cost.* On a 20-prompt serving subset a single length draw decides the ranking: Llama FP8 averages 7,288 tokens there versus 4,551 on the full grid, and timing repeats that share one sampling seed cannot expose this. As a gold-free alternative to length heuristics, unanimous agreement of just two samples lowers selective risk to 0.7–2.4%, against 2.2–5.8% for a one-sample length rule at matched coverage; five samples reach at most 0.27%.

---

## Research questions (as in `main.tex`)

1. **RQ1.** Which pass@1 differences survive an item-level interval, and how much of a 4-bit gap does maj@5 remove?
2. **RQ2.** Is a BF16-correct length increase a property of the checkpoint, or a selection effect of conditioning on quantized failures?
3. **RQ3.** At matched coverage, does k-sample answer agreement have lower selective risk than a one-sample length rule, and how does that gain scale with k?
4. **RQ4.** How much of a subset serving-cost gap is one length draw that timing repeats do not cover?

Contributions (C1–C4) answer RQ1–RQ4: one robust accuracy drop (Llama AWQ-4, concentrated at difficulty levels 3–5); a placebo-calibrated selection effect in the BF16-correct length estimator; a cost curve for gold-free agreement; subset serving cost as a length draw.

**Venue:** see [`../docs/VENUE.md`](../docs/VENUE.md) (JCR Q1 required; JSS and FGCS closed). Q1 revision notes: [`REVISION_RESPONSE_Q1.md`](REVISION_RESPONSE_Q1.md).

---

## Canonical findings (aligned with `main.tex`)

| Topic | Claim in this manuscript |
|---|---|
| Serving stack | Pinned `qrm-official` / vLLM 0.7.0 eager / A100; FP8 runs as Marlin W8A16 fallback, not native W8A8 |
| Llama AWQ-4 | Only contrast significant on MATH-500 and GSM8K (Holm-6 and Holm-18). Not detectable at difficulty levels 1–2; −2.48 / −3.75 / −4.63 pp at levels 3 / 4 / 5 |
| AWQ-4 artifacts | Properties of the `jakiAJK` uploads: own chat template (omits the `<think>\n` generation suffix; the model emits it itself), float16 activations, configured GEMM `awq` path |
| Qwen AWQ-4 GPQA | −5.56 pp; borderline under Holm-6, not significant under Holm-18; not a headline. GPQA prompts are byte-identical across checkpoints, so the contrast is paired |
| Length | Qwen 4-bit +6.3–6.9% ratio of means; the BF16-correct conditional is selection-biased (seed placebo +138 / +235); the unconditional estimator is primary |
| Pathology | 25 loop-flagged completions; 209 near-cap (≥32,500 tokens); per-item cap band (within 16 tokens of the prompt-specific cap): 155 on MATH-500, containing 139 of 140 near-cap rows |
| Modal agreement | 5/5 observed risk ≤ 0.27% (Wilson upper bounds on 0/n cells 0.82–1.08%); two-sample unanimity 0.7–2.4% vs 2.2–5.8% for the matched length rule. Not G-Pass@k, not calibration, not a safety property |
| Cost | No deployment cost ranking is claimed. Subset GPU-seconds and campaign-length seconds do not share a point order; timing repeats share one sampling seed |
| FP8 vs BF16 | 95% CIs include 0; TOST ±1 pp fails on MATH-500; not claimed equivalent |

Tables, TikZ figures, limitations, and the appendix live in `main.tex` / `main.pdf` (25 pages). Reproduce numbers with:

```bash
python3 scripts/analysis/revision_reanalysis.py --check
python3 scripts/analysis/emit_major_revision_tables.py --check
python3 scripts/analysis/revision_sensitivities.py --check
python3 scripts/check_tex_tables.py --check
python3 scripts/analysis/q1_revision_analyses.py   # levels, cap band, template and GPQA audits
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
