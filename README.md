# Estimand Sensitivity in the Evaluation of Quantized Reasoning Checkpoints

[![CI](https://github.com/Manish06N/reasoning-compression-lab/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Manish06N/reasoning-compression-lab/actions/workflows/ci.yml)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Paper and data: CC BY 4.0](https://img.shields.io/badge/paper%20%26%20data-CC%20BY%204.0-green.svg)](NOTICE.md)

Code, data and manuscript for *Estimand Sensitivity in the Evaluation of Quantized Reasoning Checkpoints: Selection Bias, Seed Placebos, and Subset-Cost Instability* (Nandish, Misra, Janarthanan; IIT Patna / Lincoln University College).

The study evaluates eight public DeepSeek-R1-Distill checkpoints (Qwen-7B and Llama-8B in BF16, FP8 run as Marlin W8A16, community AWQ-4, and RedHatAI GPTQ-4) on MATH-500, GSM8K and GPQA-Diamond. Everything runs on **one pinned stack** (vLLM 0.7.0, eager mode, one NVIDIA A100-80GB node type): 88 runs, 56,408 completions. The unit of analysis is *public checkpoint x pinned serving stack x evaluation target*; the paper makes no claim about quantization methods in general.

## Main results

1. **Accuracy.** Only Llama AWQ-4 shows a drop that survives item-level intervals and multiplicity control (MATH-500 -2.76 pp, GSM8K -1.57 pp). The tested AWQ uploads are confounded with chat template, activation dtype and backend, so this is a statement about those artifacts. Strict-majority voting over five samples lowers the point estimate to -1.4 pp, which is not itself a significant reduction.
2. **Length.** The BF16-correct token-inflation contrast is biased by selection: comparing two BF16 seeds, with no quantization, already gives +138 (Qwen) and +235 (Llama) tokens.
3. **Gold-free abstention.** Unanimity of two samples has 0.7-2.4% selective risk against 2.2-5.8% for a one-sample length rule at matched coverage (a one-sample rule costs one generation, unanimity of two costs two).
4. **Serving cost.** On a 20-prompt subset a single length draw can reorder the cost ranking; repeats that share one sampling seed cannot expose it.

Full details, limitations and threats to validity are in [`paper/main.pdf`](paper/main.pdf).

## Verify the numbers yourself (about 5 minutes, CPU only, standard library)

Requires Python 3.10 or newer (no packages needed for these checks).

```bash
git clone https://github.com/Manish06N/reasoning-compression-lab && cd reasoning-compression-lab
python3 scripts/analysis/revision_reanalysis.py --check               # pass@1, intervals, Holm, token strata
python3 scripts/analysis/emit_major_revision_tables.py --check        # generated tables vs frozen tables
python3 scripts/analysis/measured_serving_confirmation_analysis.py --check   # serving tables from 60 raw timing files
python3 scripts/analysis/modal_agreement_analysis.py --check-artifact # answer-agreement artifact
python3 scripts/check_tex_tables.py --check                           # every manuscript table cell vs data
python3 scripts/analysis/check_manuscript_numbers.py --check
```

Each command prints `OK`/`PASS`. [`REPRODUCE.md`](REPRODUCE.md) lists every check, the packages needed for the sensitivity analyses, and what can and cannot be reproduced without a GPU. [`docs/PAPER_TO_CODE.md`](docs/PAPER_TO_CODE.md) maps every table and figure of the paper to the script and output file that produce it.

## What is in this repository

```text
paper/        manuscript (main.tex, main.pdf, references.bib), arXiv source zip, artifact notes
results/      released per-run records and analysis reports
  math500/ gsm8k/ gpqa/            compact per-completion records (correctness, token count, flags)
  recovered/                       recovered MATH-500 answer strings (for answer-agreement analyses)
  measured_serving_confirmation/   raw timing JSON of the controlled serving confirmation
  reports/                         canonical analysis outputs and the runtime manifest
scripts/
  analysis/    all analyses in the paper (CPU only)
  hpc/         SLURM launchers and checks for the GPU campaign (A100, PARAM Rudra)
  check_tex_tables.py              manuscript-vs-data consistency check
src/ tests/    support library for the campaign and its unit tests
configs/ slurm/ patches/           campaign configuration, job scripts, compatibility patches
docs/          protocols and method notes; docs/archive/ holds historical project notes
```

## Data availability and scope

* Released: per-completion compact records for all 88 runs, recovered MATH-500 answer strings, raw serving timing records, runtime manifest, and every analysis script.
* Not released: full chain-of-thought traces (size and benchmark-access terms) and GPQA-Diamond item text (gated dataset; never stored here). Native stop reasons and output token IDs were not stored; the paper documents what that implies.
* The GPU campaign is inspectable but not expected to be rerun: it needs an A100-80GB, vLLM 0.7.0, the listed checkpoints and the frozen environment (`requirements-qrm-paper-vllm070.lock`). Token counts and prompt lengths are reproducible only with the frozen tokenizer versions (transformers 4.47.x); `scripts/analysis/audit_checks.py` checks this.

## Citation and license

Cite via [`CITATION.cff`](CITATION.cff) or:

```bibtex
@misc{nandish2026estimand,
  title  = {Estimand Sensitivity in the Evaluation of Quantized Reasoning Checkpoints: Selection Bias, Seed Placebos, and Subset-Cost Instability},
  author = {Nandish, Manish and Misra, Rajiv and Janarthanan, Midhunchakkaravarthy},
  year   = {2026},
  url    = {https://github.com/Manish06N/reasoning-compression-lab}
}
```

Code is MIT-licensed ([`LICENSE`](LICENSE)); the paper, results and documentation are CC BY 4.0; third-party components and what is not redistributed are listed in [`NOTICE.md`](NOTICE.md). Change history: [`CHANGELOG.md`](CHANGELOG.md).
