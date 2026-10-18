# Pre-registered verdict rules for E1c, X3 and X5

Written 2026-10-18, after E1a/E1b/E2/E3/X1 were analysed on the laptop and BEFORE any E1c, X3 or X5 cell finished (jobs 110766, 110767, 110768 were pending). Do not change after seeing outputs. Estimators: cross-seed Lambda_b (all ordered pairs s != t, BF16 run as reference), pooled placebo Pi_b from the five campaign BF16 seeds, excess = Lambda_b - Pi_b, 95% percentile intervals from one shared item bootstrap (B = 10,000, default_rng(0)).

Known before E1c (from E1a, awq_marlin kernel, MATH-500, 5 seeds): accuracy -1.68 pp [-2.96, -0.40]; Lambda_b +438 [295, 587]; excess +203 [66, 341].

## E1c (Llama self-AWQ, forced AWQ GEMM kernel, MATH-500 seeds 42-46)
Question: do E1a's accuracy loss and length excess depend on the awq_marlin kernel?
1. Accuracy: kernel-robust if the 95% interval of (E1c - BF16 campaign) excludes 0; otherwise report as "not detected under GEMM".
2. Length excess: kernel-robust if the E1c excess interval excludes 0 on the positive side. Kernel-dependent if the E1c excess interval includes 0 AND the paired item bootstrap of excess(E1a) - excess(E1c) excludes 0. Otherwise inconclusive. Report all three intervals either way.
3. Do not pool E1a and E1c.

## X3 (Qwen BF16, max_num_seqs 64 vs the campaign's 256, seeds 42-46)
1. Descriptive: share of completions token-identical to the campaign run at the same seed, distribution of the first divergent token index, accuracy and mean-length differences.
2. Cross-seed excess (Lambda_b of the max_num_seqs-64 runs against the campaign BF16 reference, minus Pi_b): consistent with the placebo if its interval includes 0; "batch limit induces inflation beyond the placebo" if it excludes 0 on the positive side (then reported as a limitation of the seed placebo, not hidden).
3. Same-seed Lambda_b with flips is descriptive (coupled null; expected smaller than Pi_b).

## X5 (Qwen and Llama greedy, max_num_seqs 256 vs 64, seed 42)
1. Descriptive: share of token-identical completions, first divergence index, accuracy difference, Lambda_b of config B against config A with item bootstrap interval (the same-config greedy placebo is exactly 0 because greedy is deterministic).
2. A configuration placebo is called usable for the greedy setting if at least 20% of completions diverge between the configurations; otherwise reported as "greedy runs were largely invariant to this change; no placebo can be formed from it".
3. One seed only: no claim beyond the stated descriptive numbers.

## Reporting limits
Until these rules are applied by the laptop analysis, the HPC agent reports only: gate result, per-seed pass@1, mean completion tokens. No verdict words.
