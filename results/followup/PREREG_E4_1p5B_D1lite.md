# Message for the HPC agent: E4 at 1.5B (replaces the 14B plan) and D1-lite (external 3-bit checkpoint), with verdict rules

Written 2026-10-23, BEFORE any E4-1.5B or D1-lite cell exists. The user approves both. Apply the rules literally; do not change them after seeing outputs. Everything from earlier briefs still holds: gate with R2_EXPECT_OFFSET=1, live-engine verification written down before the first cell finishes, separate output folder per experiment, nothing deleted or moved in scratch, stop and report on any gate failure, requeue or dirty-GPU event, report only numbers computed from stored per-item records, and until I apply the rules on my laptop report only gate results, per-seed pass@1, mean completion tokens and the share of completions stopped by the length limit.

This message supersedes two parts of earlier briefs and nothing else:
- MESSAGE_X2_E4, section 2 (E4): the preferred model becomes DeepSeek-R1-Distill-Qwen-1.5B, not the 14B. X2 (AIME placebo) is unchanged.
- MESSAGE_G1_C2_D1, section D1: the W3 build plan is replaced by D1-lite below. The builders.py change, the checkpoint builds and the scratch space they needed are NOT required; do not make the builders.py change. D1 W8 stays dropped.

Priority order: E5a > G1 > E4 (1.5B) > D1-lite pilot > X2 > D1-lite seeds 43-46.

## Why (from the literature)
Lian et al. (arXiv 2606.25519) see their largest inflation on a 4B model at 3 bits (276-292%); Lotfi et al. (arXiv 2606.00206) see their largest overthinking effect on R1-Distill-Qwen-1.5B at 3-bit AWQ. Our evidence is 7-8B at 4 and 8 bits. E4 at 1.5B and D1-lite test whether the seed-placebo reading holds where the published effects are large.

## E4 at 1.5B (MATH-500, same stack)
- Model: DeepSeek-R1-Distill-Qwen-1.5B. If the weights are not available locally and cannot be downloaded within the existing rules, say so and stop; do NOT substitute another model.
- Job 110773 (the E4 pilot for the 14B): if it has not started, cancel only its E4 part and resubmit for the 1.5B; if it has started, report and let it finish. Say which.
- Arms (5 seeds 42-46 each): (i) BF16 (bfloat16), (ii) self-quantized AWQ-4 with exactly the E1 recipe (AutoAWQ 0.2.9 env, 4-bit, group size 128, zero_point=True, GEMM packing, Pile-validation calibration; float16 activations as in the pipeline), validated as for E1 (config check, 50 prompts byte-identical to the campaign's, vLLM loads it, record the resolved kernel). Campaign pipeline and prompts (doubled BOS, expected offset +1), T=0.6, top_p 0.95, 32768 tokens, own folder results/revision3/E4_1p5B, tag _e4s.
- Space: the 1.5B BF16 weights (~3.5 GB) and the AWQ checkpoint (~1.2 GB) need room. If /scratch quota blocks the download or the build, stop and tell me; do not delete or move anything.
- Pilot first: one seed per arm, project time. If an arm projects over 12 GPU-h for 5 seeds, run 3 seeds for that arm and say so (the placebo needs at least 3 BF16 seeds, 6 ordered pairs). Small models loop to the cap more often; report the cap share in the pilot.
- Rules (fixed now), same as the old E4 rules:
  1. BF16 placebo Pi_b with its 95% interval (item bootstrap, B=10,000, default_rng(0), shared resamples).
  2. Accuracy contrast self-AWQ minus BF16 (paired item bootstrap): "loss detected" if the 95% interval excludes 0, otherwise "not detected".
  3. Conditional length change Lambda_b (seed-matched and cross-seed), excess Lambda_b - Pi_b with a shared item bootstrap: "excess detected" if the cross-seed interval excludes 0, otherwise "not detected".
  4. Cap share per cell; if more than 30% of completions in an arm stop at the length limit, I will report the length analysis for that arm as truncation-dominated.
  5. Report all intervals regardless of outcome. No pooling with other models.

## D1-lite: the external 3-bit checkpoint (Qwen-7B), descriptive
- Checkpoint: workspace/models/DeepSeek-R1-Distill-Qwen-7B-GPTQ-3 (the one named in D1 Step 0: gptqmodel 2.2.0, g128, desc_act=True, sym; calibration unknown). Do not build anything.
- Step 1 (gate, before any full cell): confirm the checkpoint's tokenizer / chat template gives prompts byte-identical to the campaign's Qwen-7B BF16 prompts for 50 items (use the campaign check); if they differ, stop and report, do not change the template. Confirm vLLM 0.7.0 loads it and generates; it must run at float16 (the plain gptq kernel accepts float16 only); record the resolved quantization method and kernel.
- Cells: MATH-500, T=0.6, top_p 0.95, 32768 tokens, campaign pipeline (doubled BOS, offset +1), float16, own folder results/revision3/D1lite_gptq3, tag _d1l. Pilot = seed 42, full 500 items. Report after the pilot: gate result, pass@1, mean tokens, cap share, wall-clock; wait for my reply before seeds 43-46. If the pilot projects over 12 GPU-h for 3 more seeds, say so and I will decide the number of seeds.
- Rules (fixed now), for the seeds that exist (at least 3 are needed for any placebo statement; with fewer I will report accuracy and length only):
  1. Accuracy: W3 minus campaign BF16 (paired item bootstrap): "loss detected" if the 95% interval excludes 0; also W3 minus the E2 float16 BF16 cells (same dtype, same pipeline).
  2. Unconditional length difference and cap share per cell.
  3. Lambda_b (seed-matched and cross-seed) against campaign BF16, excess Lambda_b - Pi_b with the five-seed campaign BF16 placebo and a shared item bootstrap: "excess detected" if the cross-seed interval excludes 0, otherwise "not detected". The placebo is estimated at bfloat16; the E2 float16 BF16 cells give the dtype-matched check (descriptive).
  4. Placebo-free item-matched correct-run difference (Proposition 2 of the draft): report it.
  5. Descriptive only: place W3 next to the 4-bit float16 cells (P2 dtype controls, whose prompts carry one token fewer) and say so. Calibration is unmatched, so no claim that 3 bits as such cause an effect.

## What to send back
A new archive exports/laptop_export_<date>_v6.tar.gz (v7 if v6 exists), built as before, with for each finished experiment: compact per-item JSON (details[].row, completion_tokens, prompt_tokens, extractive_match, finish_reason), engine-settings dumps, manifests, cell_status rows, verifier addenda. Give size, file count, sha256 and the exact scp line first. Report job ids and queue reasons when you submit, and tell me on 2026-10-27 and 2026-10-29 which parts have not started.
