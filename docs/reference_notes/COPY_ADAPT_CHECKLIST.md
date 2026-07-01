# Copy/Adapt Checklist — Paper 1 Core Code

This repo contains **adapted logic only**. Full external repos stay in `../external_repos/` as read-only reference.

## What was adapted

| Source repo | Source file(s) | Our file(s) |
|-------------|----------------|-------------|
| sober-reasoning | `lighteval_tasks.py` | `prompts/math500.txt`, `prompts/gpqa_diamond.txt` |
| sober-reasoning | extraction ideas | `src/extraction/math_extractor.py`, `gpqa_extractor.py` |
| sober-reasoning | `run.sh` seed loop | `slurm/run_grid.sh` |
| sober-reasoning | mean±std reporting | `src/stats/seed_variance.py` |
| Quantized-Reasoning-Models | `inference.py` decoding | `configs/decoding/repro_qrm.yaml` |
| Quantized-Reasoning-Models | `make_stats_table.py` | `src/metrics/trace_length.py` |
| Calibrating-LLMs-with-Consistency | `consistency.py` | `src/metrics/consistency.py` |
| Calibrating-LLMs-with-Consistency | `utils.py` | `src/metrics/calibration.py`, `selective_risk.py` |
| Cost-of-Pass | `estimate.py` | `src/metrics/cost_per_correct.py`, `profiling/local_cost_model.py` |
| Cost-of-Pass | `FrontierCostofPass` | `src/metrics/pareto_frontier.py`, `scripts/build_pareto_frontier.py` |
| Calibrating-LLMs | multi-sample predict | `scripts/run_inference_multisample.py`, `scripts/score_multisample.py` |
| Quantized-Reasoning-Models | `make_stats_table.py` targets | `configs/baselines/qrm_literature_targets.yaml`, `scripts/compare_qrm_baseline.py` |
| sober-reasoning + vLLM | SamplingParams | `src/runners/sampling_utils.py`, `scripts/verify_decoding_params.py` |
| lm-evaluation-harness | optional sanity | `scripts/lmeval_sanity_check.sh`, `docs/reference_notes/LMEVAL_SANITY.md` |
| Quantized-Reasoning-Models | HF GPTQ weights | `scripts/hpc/08_download_gptq4_models.sh` |

## What was NOT copied

- Quantized-Reasoning-Models `methods/`, `third-party/`, forked vLLM
- Full `lighteval_tasks.py` or lm-eval harness
- AbstentionBench pipeline (Phase 2+)
- GPTQ/SmoothQuant executable code (install via pip / use checkpoints)

## HPC sync rule

Sync **only** `reasoning-compression-lab/` to HPC. Keep `external_repos/` on MacBook unless debugging quant install.

## Pipeline

```text
runs/raw/        scripts/run_inference.py
runs/extracted/  scripts/extract_answers.py
runs/scored/     scripts/score_run.py
results/         summaries + scripts/compute_calibration.py (pilot)
```
