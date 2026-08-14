Core Paper 1 literature groups:

**Updated 2026-08-14:** the local merged bundle is incomplete for the current novelty decision. Add the primary sources below to the supervisor/manuscript review even if they are not yet present in `ALL_PAPERS_MERGED.md`.

1. Quantized reasoning baseline
- Quantization Hurts Reasoning?
- Quantization Meets Reasoning

2. Primary quantization methods
- GPTQ
- AWQ
- SmoothQuant
- LLM.int8 / ZeroQuant as background

3. Calibration / confidence
- Calibrating LLMs with Sample Consistency
- Unsupervised Confidence Calibration for Reasoning LLMs
- Reasoning Models Better Express Their Confidence
- Just Ask for Calibration

4. Reproducibility / seed variance
- A Sober Look at Progress in Language Model Reasoning

5. Cost / efficiency
- Cost-of-Pass
- OckBench
- From Prompts to Power

6. Abstention / selective prediction
- AbstentionBench
- Know Your Limits

7. Current quantized reliability and failure modes
- Quantized LLMs Can Still Be Calibrated (ACL 2025): https://aclanthology.org/2025.acl-long.1473/
- Reliability Scaling Laws for Quantized LLMs (2026): https://arxiv.org/abs/2607.10855
- Quantization Inflates Reasoning (2026): https://arxiv.org/abs/2606.25519
- Extreme Low-Bit Failure Modes (2026): https://arxiv.org/abs/2606.02011
- Calibrated e-CUSUM for quantized reasoning degeneration (2026): https://arxiv.org/abs/2607.11317
- BitCal-TTS (2026): https://arxiv.org/abs/2605.05561

## Where to read

| Group | PDF source | Code reference |
|-------|------------|----------------|
| 1. Quantized reasoning | [paper1/ALL_PAPERS_MERGED.pdf](paper1/ALL_PAPERS_MERGED.pdf) · [text extract](paper1/ALL_PAPERS_MERGED.md) | `external_repos/01-core-baselines/Quantized-Reasoning-Models/` |
| 2. Quantization methods | same bundle (GPTQ, SmoothQuant, …) | `external_repos/03-quantization-implementations/` |
| 3. Calibration | merged bundles | `external_repos/02-calibration-and-cost/Calibrating-LLMs-with-Consistency/` |
| 4. Seed variance | merged bundles | `external_repos/01-core-baselines/sober-reasoning/` |
| 5. Cost | merged bundles | `external_repos/02-calibration-and-cost/Cost-of-Pass/` |
| 6. Abstention | merged bundles | `external_repos/05-selective-prediction/AbstentionBench/` |

## Reading order (Week 1)

1. Quantization Hurts Reasoning? (paper + skim reference repo)
2. A Sober Look at Progress in Language Model Reasoning
3. Calibrating LLMs with Sample Consistency
4. Cost-of-Pass
5. GPTQ paper (primary method — in merged bundle)

After these five, read the current quantized-reliability/failure-mode group before freezing novelty. Do not run a broad grid in parallel; only recovery Phase 0 and approved matched pilots may proceed.
