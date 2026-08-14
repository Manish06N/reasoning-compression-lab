# Working Manuscript — Not Submission Ready

**Status (2026-08-14): Needs revision.** This file is now a claim-evidence outline, not a finished manuscript. The current FP8 runs are replication/control evidence only. Do not write a causal quantization, calibration, or performance conclusion until the gates in [the publication-recovery plan](../docs/plans/2026-08-14-publication-recovery.md) pass.

**Canonical audit:** [Publication Readiness Audit](../docs/PUBLICATION_READINESS.md)

# Abstract

To be written after the contribution-selection gate. It must state the matched design, number of seeds, primary endpoints, controlled hardware/stack, and limitations. Do not lead with the provisional single-seed FP8 values.

# 1. Introduction

- Problem: pass@1 alone hides termination, degeneration, confidence, and deployment-cost behavior.
- Existing work already covers broad quantized-reasoning accuracy, calibration under quantization, and cost-per-pass; the contribution must be narrower.
- Candidate gap: paired reliability–cost behavior under quantization and controlled serving-stack shift.
- Contributions will be finalized only after the discriminating pilot.

# 2. Related Work

## Quantized reasoning models

Position against QRM and later low-bit reasoning/failure-mode studies; do not claim the first systematic quantized-reasoning evaluation.

## Calibration and uncertainty

Position against comprehensive calibration-under-quantization work; emphasize reasoning traces and validated confidence construction only if supported.

## Selective prediction and abstention

## Cost-aware model evaluation

Distinguish expected cost per correct answer from raw wall time and list every pricing/energy assumption.

## Reproducibility and seed variance

Motivate matched prompts, decoding, stack versions, seeds, and trace-level provenance.

# 3. Method

## Models

Headline anchors: DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B. Record exact artifact revisions, not only display names.

## Quantization settings

BF16, FP8-checkpoint/Marlin fallback on A100, AWQ4, and GPTQ4. Never label the observed A100 path native FP8 W8A8 execution.

## Tasks

MATH-500 first; GPQA-Diamond/GSM8K only after the contribution gate. Pin dataset revision and fingerprint.

## Decoding protocol

Protocol P1-2026-08: identical prompt, engine, scheduler, max length, temperature, top-p, penalties, and seeds 42–46 across compared formats.

## Metrics

pass@1, maj@5 subset, cap/loop/parse rates, Brier/ECE/AURC with valid confidence, latency distribution, peak VRAM, energy availability, and cost-per-correct.

## Statistical testing

Paired McNemar, paired bootstrap, seed-level variation, Holm correction, and predeclared primary comparisons.

## Hardware and serving setup

Report A100 node, GPU allocation, stack commit/version, memory target, warm-up, scheduler preemptions, and whether a measurement is offline correctness or controlled serving telemetry.

# 4. Results

## Provisional replication result — appendix/control only

| Model | Setting | MATH-500 pass@1 | Interpretation |
|-------|---------|----------------:|----------------|
| Qwen-7B | FP8 checkpoint, QRM stack, seed 42 | 472/500 (94.4%) | Compatible with existing FP8 model-card value; no matched BF16 |
| Llama-8B | FP8 checkpoint, QRM stack, seed 42 | 445/500 (89.0%) | Compatible with existing FP8 model-card value; no matched BF16 |

Six likely near-cap traces and phrase-level degeneration motivate improved instrumentation. These observations are provisional because `finish_reason` and token IDs were not saved.

## Accuracy under compression

Blocked until matched BF16/FP8/AWQ4/GPTQ4 cells complete.

## Calibration under compression

Blocked until valid confidence rows and the maj@5 subset complete.

## Selective risk

Blocked until calibration provenance passes.

## Seed variance

Blocked until the three-seed pilot; headline claims require five seeds.

## Cost-per-correct

Blocked until controlled timing and explicit cost assumptions are available.

## Latency and VRAM

Blocked. Slurm elapsed time for jobs 96100/96101 is not comparable because Llama incurred more than 900 scheduler preemptions/recomputations.

## Trace behavior

Report finish reasons, cap rates, phrase/n-gram loops, parse failures, and length distributions after schema repair.

# 5. Discussion

Separate quantization effects from serving-stack effects and distinguish statistical from deployment significance.

# 6. Limitations

Must cover A100 FP8 fallback, benchmark/task scope, confidence construction, shared-cluster measurement limits, model-family scope, and stack specificity.

# 7. Conclusion

Write only after the primary contribution and all supported claims are frozen.
