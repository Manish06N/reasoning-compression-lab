# Beyond Pass@1: Reliability and Token-Cost Effects of Quantized Reasoning Models under a Pinned Serving Stack

**Manish Nandish**
*Department of Computer Science & Engineering, Indian Institute of Technology Patna*
*PARAM Rudra HPC (NSM / C-DAC)*
Email: manishn_iitp@iitp.ac.in

**Do not cite this markdown file for numbers.** The canonical manuscript is [`main.tex`](main.tex) compiled to [`main.pdf`](main.pdf). This file exists only so repository markdown matches the LaTeX story.

**Keywords:** Reasoning language models, post-training quantization, pinned serving stack, token inflation, selective prediction, Cost-of-Pass.

---

## Abstract (same claims as `main.tex`)

Post-training quantization is a common serving default for reasoning language models, but pass@1 on a single seed can hide changes in trace length, termination behavior, and sample-level reliability. We pin one serving stack—vLLM 0.7.0 eager execution on an NVIDIA A100-80GB, with FP8 checkpoints executed as Marlin W8A16 rather than native W8A8—and vary only the weight checkpoint among BF16, FP8, AWQ-4, and GPTQ-4 for DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B. The grid has 88 cells and 56,408 completions on MATH-500 (5 seeds), GSM8K (3 seeds), and GPQA-Diamond (3 seeds).

On MATH-500, FP8–BF16 pass@1 differences are $+0.40$ and $+0.28$ percentage points (pp). Problem-clustered bootstrap 95% intervals include zero; a $\pm 1$ pp TOST equivalence test is **not** passed. Four-bit effects are architecture- and task-dependent: Llama AWQ-4 loses $2.76$ pp on MATH-500 (95% CI $[-4.16,-1.44]$, $p<0.001$) and $1.57$ pp on GSM8K; Qwen AWQ-4 loses $5.56$ pp on GPQA-Diamond (95% CI $[-9.60,-1.52]$, $p=0.007$). Majority-vote McNemar tests on maj@5 remain non-significant and do not answer the pass@1 question. Mean MATH-500 length rises $6.3$–$6.8\%$ for Qwen 4-bit (paired token CIs exclude zero); the increase is concentrated on items where BF16 is correct and the quantized run is not. Identical-word loops are rare but nonzero ($25/56{,}408$); none of the rows hit the $32{,}768$ token cap after re-encoding, yet $209$ completions sit at $\ge 32{,}500$ tokens. Using recovered answer strings, gold-free modal agreement is an observable abstention signal: 5/5 consensus has very low selective error on MATH-500 but costs five generations per query, and Llama AWQ-4 reduces 5/5 coverage by $6.0$ pp versus BF16 (95% paired CI $[-9.4,-2.6]$). Cost-of-Pass is a token ranking at a shared $65$ tok/s assumption, not measured wall-clock.

---

## Research questions

1. **RQ1.** Does quantization change *pass@1* relative to matched BF16, with problem-clustered uncertainty?
2. **RQ2.** Do identical-word loops and near-budget terminations occur, and do they differ by format?
3. **RQ3.** What can be said about multi-sample reliability without gold labels at serve time?
4. **RQ4.** How does output length change a *token-implied* Cost-of-Pass when throughput is held fixed?

This paper **pins** one stack. It does not run a factorial vLLM 0.7.0 vs 0.8.5 experiment and does not claim a Serving-Stack Shift result.

---

## Canonical findings (aligned with `main.tex`)

| Topic | Claim in this manuscript |
|---|---|
| Serving stack | Pinned `qrm-official` / vLLM 0.7.0 eager / A100 W8A16 FP8 fallback |
| Pathology | 25 loop-flagged completions; 0 exact cap hits; 209 near-cap ($\ge 32{,}500$ tokens) |
| Llama AWQ-4 | Significant MATH-500 and GSM8K pass@1 degradation vs BF16 |
| Qwen AWQ-4 | Significant GPQA-Diamond pass@1 degradation vs BF16 |
| Qwen 4-bit tokens | About $+6$–$7\%$ mean length vs BF16 on the full MATH-500 grid |
| 200-item subset | Retracted as a result (even-index / seed-42 estimator artifact) |
| Modal-answer selective prediction | Gold-free MATH-500 modal agreement; 5/5 selective error $\le 0.27\%$ in this sample; Llama AWQ-4 5/5 coverage $-6.0$ pp vs BF16. Not a safety property. |
| Gold-hit 98.23% “safety gate” | Removed; not an operational abstention rule |
| Cost | Fixed-throughput token-cost proxy; not Pareto-optimal; not measured tok/s |
| FP8 vs BF16 | 95% CIs include 0; **not** claimed as “FP8 matches BF16” or $\pm 1$ pp equivalent |

Tables, TikZ figures, limitations, and the appendix live in `main.tex` / `main.pdf`. Reproduce numbers with:

```bash
python3 scripts/analysis/revision_reanalysis.py --check
python3 scripts/analysis/modal_agreement_analysis.py --check
```

## Artifacts

https://github.com/Manish06N/reasoning-compression-lab

Per-cell records: `results/math500/`, `results/gsm8k/`, `results/gpqa/`. Canonical report: `results/reports/revision_reanalysis_report.json`. Modal agreement: `results/reports/modal_agreement_report.json`. Packaging: [`ARTIFACT.md`](ARTIFACT.md).
