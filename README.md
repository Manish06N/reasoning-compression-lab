# reasoning-compression-lab

Deployment-science evaluation harness for compressed reasoning LLMs.

**Paper 1:** *Beyond Accuracy: Reliability, Calibration, Seed Variance, and Cost-per-Correct of Quantized Reasoning LLMs*

## Research question

When reasoning LLMs are compressed, do they remain **accurate, calibrated, stable, fast, memory-efficient, and economically useful** under real serving conditions?

## Division of labor

| Machine | Role |
|---------|------|
| **MacBook** | Design docs, scripts, configs, writing, plotting from CSVs, log review |
| **HPC / A100** | Model download, vLLM inference, quantization, main experiments, profiling |

## Repository layout

```
configs/          Experiment cell, task, model, and quantization configs
prompts/          Task prompts and templates
scripts/          CLI entrypoints and batch helpers
src/              Runners, metrics, extraction, stats, profiling, quantization
runs/             Raw → extracted → scored pipeline outputs
results/          Aggregated CSVs and tables for paper
paper_figures/    Final publication figures
paper/            Manuscript draft
docs/             Design, runbook, experiment log
slurm/            HPC job scripts
notebooks/        Analysis dashboards
```

## Execution gates

1. **Level A — reproduction:** Qwen-7B × {BF16, GPTQ-4} × MATH-500 × seed 0
2. **Level B — pilot:** Qwen-7B × 5 configs × {MATH-500, GPQA-Diamond} × 5 seeds
3. **Level C — main grid:** 3 models × 5 configs × 4 tasks × 5 seeds

No full grid before reproduction works. No 14B before Qwen-7B pilot works.

## First experiment

```
Model:  DeepSeek-R1-Distill-Qwen-7B
Task:   MATH-500
Config: BF16
Seed:   0
Hardware: A100
```

## Reference repos

Cloned under `../external_repos/` for reading only — do not develop inside them.

- [Quantized-Reasoning-Models](https://github.com/ruikangliu/Quantized-Reasoning-Models)
- [sober-reasoning](https://github.com/bethgelab/sober-reasoning)
- [Calibrating-LLMs-with-Consistency](https://github.com/veronica320/Calibrating-LLMs-with-Consistency)
- [Cost-of-Pass](https://github.com/mhamzaerol/Cost-of-Pass)
- [AbstentionBench](https://github.com/facebookresearch/AbstentionBench)

## Docs

- [PAPER1_DESIGN.md](docs/PAPER1_DESIGN.md) — scope, models, tasks, metrics, claim
- [RUNBOOK.md](docs/RUNBOOK.md) — MacBook ↔ HPC workflow
- [EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) — dated experiment record
