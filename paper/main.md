# Beyond Pass@1: Accuracy, Agreement, and Serving-Cost Effects of Public R1-Distill Quantization Checkpoints under a Pinned Stack

**Manish Nandish**
*Department of Computer Science & Engineering, Indian Institute of Technology Patna*
*PARAM Rudra HPC (NSM / C-DAC)*
Email: manishn_iitp@iitp.ac.in

**Do not cite this markdown file for numbers.** The canonical manuscript is [`main.tex`](main.tex) compiled to [`main.pdf`](main.pdf). This file exists only so repository markdown matches the LaTeX story.

**Keywords:** Reasoning language models, public quantization checkpoints, pinned serving stack, estimand disagreement, Cost-of-Pass.

---

## Abstract (same claims as `main.tex`)

Post-training quantization is a common serving default for reasoning language models, but a single pass@1 or token count is not a complete evaluation. We pin one serving stack—vLLM 0.7.0 eager execution on an NVIDIA A100-80GB, with FP8 checkpoints executed as Marlin W8A16 rather than native W8A8—and vary only the public weight checkpoint among BF16, FP8, AWQ-4, and GPTQ-4 for DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B. The grid has 88 cells and 56,408 completions on MATH-500 (5 seeds), GSM8K (3 seeds), and GPQA-Diamond (3 seeds).

Under this pin, estimands disagree. On MATH-500, FP8–BF16 pass@1 differences are $+0.40$ and $+0.28$ pp; clustered 95% intervals include zero, and $\pm 1$ pp TOST is not passed. The tested `jakiAJK` Llama AWQ-4 checkpoint loses $2.76$ pp on MATH-500 and $1.57$ pp on GSM8K; the tested Qwen AWQ-4 checkpoint loses $5.56$ pp on GPQA-Diamond. maj@5 McNemar is a different consensus estimand. Mean MATH-500 length rises $6.3$–$6.9\%$ for Qwen 4-bit; Both-OK CIs exclude zero for Qwen AWQ-4/GPTQ-4 while BF16-only mismatches are thousands of tokens longer. Loops: $25/56{,}408$. No row re-encoded to $\ge 32{,}768$; native stop reasons were not logged; $209$ near-cap completions. Gold-free unique-mode abstention on five MATH seeds has low observed 5/5 selective error; Wilson intervals on $0/n$ cells remain strictly positive. Token-proxy, sequential, and batched GPU-second rankings disagree. Qwen FP8 Condition B is bimodal across five repeats and is not a lone $-36.0\%$. Llama GPTQ-4 Condition B mean throughput is within $0.2\%$ of Llama BF16.

---

## Research questions

1. **RQ1.** Does quantization change *pass@1* relative to matched BF16, with problem-clustered uncertainty?
2. **RQ2.** Do identical-word loops and near-cap completions occur, and do they differ by checkpoint?
3. **RQ3.** What can be said about multi-sample reliability without gold labels at serve time?
4. **RQ4.** Do format rankings agree across a token-proxy Cost-of-Pass, sequential GPU-seconds, and batched GPU-seconds?

This paper **pins** one stack. It does not run a factorial vLLM 0.7.0 vs 0.8.5 experiment.

---

## Canonical findings (aligned with `main.tex`)

| Topic | Claim in this manuscript |
|---|---|
| Serving stack | Pinned `qrm-official` / vLLM 0.7.0 eager / A100 W8A16 FP8 fallback |
| Pathology | 25 loop-flagged completions; 0 exact cap hits; 209 near-cap completions ($\ge 32{,}500$ tokens); Qwen AWQ/GPTQ MATH near-cap $25$/$24$ vs BF16 $14$ |
| Llama AWQ-4 | Tested `jakiAJK` checkpoint: significant MATH-500 and GSM8K pass@1 drop vs BF16 |
| Qwen AWQ-4 | Tested checkpoint: significant GPQA-Diamond pass@1 drop vs BF16 (largest accuracy effect) |
| Qwen 4-bit tokens | $+6.3$–$6.9\%$ RoM vs BF16; Both-OK CIs exclude 0; Lian BF16-correct $\Delta$ positive |
| 200-item subset | Superseded estimator (Appendix); not a result |
| Modal-answer selective prediction | Gold-free unique-mode abstention; 5/5 observed risk $\le 0.27\%$; Wilson upper bounds on $0/n$ cells $0.79$–$1.08\%$. Not G-Pass@k. Not a safety property. |
| Cost | Hybrid scenario $C_{\mathrm{pass}}$: confirmation GPU-sec / campaign MATH pass@1. Rankings disagree across 65 tok/s proxy, Condition A, and Condition B. Qwen FP8 B: five-rep listing, not a lone $-36.0\%$. |
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

Per-cell records: `results/math500/`, `results/gsm8k/`, `results/gpqa/`. Canonical report: `results/reports/revision_reanalysis_report.json`. Modal agreement: `results/reports/modal_agreement_report.json`. Packaging: [`ARTIFACT.md`](ARTIFACT.md).
