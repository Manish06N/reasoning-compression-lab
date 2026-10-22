# E4 at 1.5B, LOCAL REPLICATE on a second stack: protocol and verdict rules (written 2026-10-23, BEFORE any local cell exists)

Status: SECONDARY. The primary E4 is cluster job 111867 (pinned stack), whose rules are in MESSAGE_E4small_D1lite_for_HPC_agent.md. This file defines an optional replicate on the user's local GPU; it is run only if the user approves it. It uses the AWQ-4 checkpoint built by job 111867 (same recipe as E1), shipped to this machine. The replicate's result never replaces the primary one; if both exist, both are reported.

Model: DeepSeek-R1-Distill-Qwen-1.5B. Arms: (i) BF16, (ii) the AWQ-4 checkpoint built by job 111867 on the cluster with exactly the E1 recipe (AutoAWQ 0.2.9, 4-bit, g128, zero_point=True, GEMM packing, Pile-validation calibration), shipped to this machine. Benchmark: MATH-500 (500 items), seeds 42-46, 5 seeds per arm.

## Where it differs from the campaign (stated as deviations, not hidden)
- Generation runs on a local RTX 5080 (16 GB, Blackwell) in WSL, with vLLM 0.23.0 (V1 engine) and torch 2.11, not vLLM 0.7.0 eager on an A100. The campaign BF16 cells of the 7-8B models are NOT used as a reference; E4 is a self-contained comparison of the 1.5B model with itself (BF16 seeds against BF16 seeds, AWQ-4 against BF16), both arms on the same local stack. No pooling with other models.
- Prompts: the campaign prompt template (prompts/qrm_math500.txt) applied by the local runner. The campaign pipeline reaches the engine with a doubled BOS token; the local runner will state which it uses (it uses ONE BOS, from the tokenizer, for both arms), and the same prompt tokens go to both arms. The effect of the extra BOS is tested separately (E5a) and is not part of E4.
- Sampling: T=0.6, top_p 0.95, max 32768 new tokens, seeds 42-46 passed as per-request seeds, max_model_len 32768. Prefix caching and chunked prefill off where the engine allows it; the effective engine settings are dumped to disk per cell.
- Scoring: the campaign's scoring functions (LightEval 0.8.0 metric objects, extractive match), run in a separate CPU environment on the stored completions, after checking that it reproduces the stored `extractive_match` of the campaign on 500/500 Qwen-7B MATH-500 completions' answer strings where available; if that check cannot be run, the scorer is math-verify 0.9.0 with the same extraction configuration and this is stated.
- A different serving stack and GPU change the numerical noise; that is a feature for a placebo (the placebo is estimated on this stack) and a limitation for comparability with the 7-8B results. The paper will say so.

## Gates (before any cell is counted)
1. Stored per-cell record: engine settings dump (vLLM version, dtype, quantization method and kernel actually resolved, max_num_seqs, enable_prefix_caching, chunked prefill, seed handling).
2. BF16 sanity: pass@1 of the first BF16 cell within 78-88% (published R1-Distill-Qwen-1.5B MATH-500 results are in the low 80s); if outside, stop and debug the pipeline, do not continue to other seeds.
3. AWQ checkpoint: config check, loads, resolved kernel recorded, first 50 prompts identical to the BF16 arm's prompts (same template, same tokenizer).
4. Per-item records use the campaign compact schema: `details[].row` (1-based dataset row), `completion_tokens`, `prompt_tokens`, `extractive_match`, `finish_reason`.

## Verdict rules (fixed now; item bootstrap B=10,000, numpy default_rng(0), shared resamples across the quantities compared)
1. BF16 seed placebo Pi_b (all ordered cross-seed pairs, closed form) with its 95% interval.
2. Accuracy contrast AWQ-4 minus BF16 (paired item bootstrap): "loss detected" if the 95% interval excludes 0, otherwise "not detected".
3. Conditional length change Lambda_b (BF16-correct conditioning; seed-matched and cross-seed pairs) and excess Lambda_b - Pi_b: "excess detected" if the cross-seed interval excludes 0, otherwise "not detected". Also report the placebo-free item-matched correct-run difference (Proposition 2 of the draft).
4. Cap share per cell. If more than 30% of the completions of an arm stop at the length limit, the length analysis for that arm is reported as truncation-dominated, and Lambda_b is reported together with the finished-only version.
5. Report all intervals regardless of outcome. If fewer than 5 seeds finish for an arm, the number is stated; at least 3 BF16 seeds are needed for any placebo statement.
