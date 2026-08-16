# Paper 1 Design — controlled revision

> **Superseding note (2026-08-16):** Canonical title is *…under a Pinned Serving Stack* (`paper/main.tex`). This file's “Serving-Stack Shift” title is historical.

**Status (2026-08-14): Needs revision; design is not frozen for submission.**

**Evidence:** [PUBLICATION_READINESS.md](PUBLICATION_READINESS.md) · **Execution:** [plans/2026-08-14-publication-recovery.md](plans/2026-08-14-publication-recovery.md)

**Full thesis roadmap:** [PHD_ROADMAP.md](PHD_ROADMAP.md)

## Provisional title

*Beyond Pass@1: Reliability–Cost Frontiers of Quantized Reasoning Models under Controlled Serving-Stack Shift*

The previous broad title remains a useful theme, but is not itself a novelty claim.

## Primary question

When model, task, prompt, decoding, seed, engine, and hardware are matched, how do weight quantization and serving-stack choices affect correctness, termination/degeneration, calibrated selective risk, and cost per correct reasoning answer?

## Contribution gate

The three-seed discriminating pilot selects one primary track:

| Track | Proceed only if |
|-------|-----------------|
| **A. Quantization reliability–cost frontier** | BF16/FP8/AWQ4/GPTQ4 differences persist across seeds in reliability, calibration, or cost—not just pass@1 |
| **B. Controlled serving-stack transfer** | Stack effects dominate and can be isolated one factor at a time |
| **Negative/replication artifact** | Neither effect survives matched controls and audit |

No title or abstract may imply Track A or B before this gate.

## Frozen pilot scope

| Dimension | Pilot | Confirmatory target after contribution gate |
|-----------|-------|--------------------------------------------|
| Models | Qwen-7B, Llama-8B | Same; no new family before approval |
| Formats | BF16, FP8-checkpoint/Marlin, AWQ4, GPTQ4 | Same; GPTQ3 appendix only if predeclared |
| Task | MATH-500 | Add GPQA-Diamond/GSM8K only if justified |
| Seeds | 42, 43, 44 | 42–46 for headline MATH-500 cells |
| Completions | pass@1 plus predeclared maj@5 subset | Same confidence construction across formats |
| Hardware | PARAM Rudra A100 | Controlled, nonexclusive one-GPU cells with explicit telemetry |

Qwen-1.5B, 14B+, Qwen3, GGUF, KV-cache quantization, and LiveCodeBench are gated extensions, not part of the pilot.

## Required endpoints

- Correctness: pass@1 and a predeclared maj@5 subset.
- Reliability: `finish_reason`, cap/stop rate, repeated n-grams/phrases, parse success, completion length.
- Calibration/selective risk: Brier, ECE, AURC using a validated confidence source.
- Stability: seed-level variation and rank reversals.
- Systems: per-request latency, throughput, peak VRAM, measured energy when available.
- Economics: cost-per-correct with explicit assumptions; Joules-per-correct only with valid telemetry.

## Statistical plan

- Paired McNemar tests and paired problem bootstrap for correctness contrasts.
- Seed variation reported separately from problem-level sampling uncertainty.
- Holm correction over predeclared primary comparisons.
- At least 200 stratified manual traces plus every flagged cap/loop/parse case.
- Negative and null findings retained.

## Current evidence boundary

Jobs 96100/96101 are single-seed FP8 replication/control rows: Qwen 94.4% and Llama 89.0% on MATH-500. They cannot establish a quantization effect, calibration, native FP8 performance, or cost. They may appear in a replication appendix after provenance repair.

## Superseded rule

The former statement that the seed-0 b01–b09 grid could be the “first publishable core” is withdrawn. A single seed is pilot evidence only, and the old main-grid protocol must not be mixed with the exact-QRM protocol.
