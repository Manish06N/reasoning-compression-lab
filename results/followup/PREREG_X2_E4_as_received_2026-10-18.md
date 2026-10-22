# X2 (AIME placebo) and E4 (third model size) - rules as received from the user, 2026-10-18/19, BEFORE any X2 or E4 cell exists. Apply literally; do not change after seeing outputs.
Recorded by the HPC agent at 2026-10-18T21:41:42+05:30. Report only gate results, per-seed pass@1 and mean completion tokens until the user applies the rules on the laptop.

## X2: placebo on a hard benchmark (BF16 only)
Models: DeepSeek-R1-Distill-Qwen-7B and -Llama-8B, BF16, campaign config (vLLM 0.7.0 eager, T=0.6, top_p=0.95, max tokens 32768, max_num_seqs 256).
Data: AIME-2024 and AIME-2025 (60 problems), benchmark definitions in external/Quantized-Reasoning-Models; record dataset revision and item order table (row, problem id, year).
Seeds: 42-49 (8) after a PILOT of one seed per model; project total; if projection > 6 GPU-h per model run 6 seeds (42-47) and say so; never fewer than 4.
Rules (descriptive): placebo Pi_b (all ordered seed pairs, item bootstrap B=10,000, default_rng(0)) with 95% interval per family, as tokens and as % of mean completion length (MATH-500 reference 3.4% Qwen, 5.1% Llama). Also: share of intermediate items (solved in some but not all seeds), share of completions in the near-cap band (>=32,500 tokens), unconditional mean length, NDR flip placebo. Placebo "detected" for a family if its interval excludes 0, else "not detected at n=60". No further verdict.
Pre-stated expectation (not a rule): placebo larger as a share of mean length than on MATH-500.

## E4: a third model size
Preferred: DeepSeek-R1-Distill-Qwen-14B (found complete in ~/models_offload; read-only use). If not available use Qwen-1.5B. Do NOT switch to Qwen3-8B without asking.
Arms (MATH-500, seeds 42-46): (i) BF16 (bfloat16); (ii) self-AWQ-4 with exactly the E1 recipe (AutoAWQ 0.2.9, 4-bit, g128, zero_point=True, GEMM, Pile-val calibration), validated as for E1 (config check, 50 prompts byte-identical, vLLM loads it), record the resolved kernel, runs at float16.
Pilot first: one seed per arm, project time; if an arm projects over 12 GPU-h for 5 seeds run 3 seeds for that arm and say so (placebo needs >=3 BF16 seeds).
Rules (descriptive): BF16 placebo Pi_b with 95% interval. Accuracy contrast self-AWQ minus BF16 (paired item bootstrap): "loss detected" if the 95% interval excludes 0 else "not detected". Conditional length change Lambda_b (seed-matched and cross-seed), excess Lambda_b - Pi_b with shared item bootstrap: "excess detected" if the interval excludes 0 on either side else "not detected". Report all three intervals regardless. No pooling with other models.

## Order and limits
Submit pilots for X2 and E4 (nice, after E1c/X3/X5 are queued); full runs only after the pilots are projected and reported. Nothing else (no E5a, no extra arms). Gate failure -> stop and report. Gate with R2_EXPECT_OFFSET=1, live-engine verification, separate output folders per experiment, nothing deleted or moved in scratch.

---
# Engine/gate expectations for X2 and E4 pilots (written 2026-10-18T21:41:42+05:30, before submission)
| Group | Expected effective values |
|---|---|
| X2 pilot (cells/x2_pilot.tsv; outputs results/revision3/X2_aime24_pilot and X2_aime25_pilot, one folder per dataset) | dtype bfloat16, quantization None, max_num_seqs 256, eager, max_model_len 32768, prefix/chunked off, T 0.6, top_p 0.95, 32768 tokens, seed 42 |
| E4 BF16 pilot (cells/e4_bf16.tsv -> results/revision3/E4_14B_bf16_pilot) | same, model ~/models_offload/DeepSeek-R1-Distill-Qwen-14B (tokenizer.json/tokenizer_config.json/generation_config.json byte-identical to the 7B model, sha256 checked) |
| E4 AWQ pilot (cells/e4_awq.tsv -> results/revision3/E4_14B_awq_pilot) | dtype float16, quantization 'awq' or 'awq_marlin' (recorded), max_num_seqs 256, rest as above; checkpoint validated by validate_checkpoint_e4.py (VALIDATE_OK required) |
First-cell gate: runs only for MATH-500 n=500 (E4 cells) with R2_EXPECT_OFFSET=1; there is no campaign reference for AIME, so the X2 prompt-token offset (+1 expected) is checked afterwards on CPU from the stored sidecar and reported. DIR_COLLISION guard active. Exit 3/4, requeue or dirty-GPU event: stop and report.
Datasets: AIME_2024 = Maxwell-Jia/AIME_2024 rev 8d88b2876a82a080e2f172cc9b25d0d9d2cb4792 (30 rows); aime_2025 = yentinglin/aime_2025 rev 6f71d77b0b89b9dabe07ab466c51df33f514df7f (30 rows); saved with datasets.save_to_disk into the WORKSPACE copy external/Quantized-Reasoning-Models/datasets (frozen repo untouched).
