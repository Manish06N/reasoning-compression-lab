# Message for the HPC agent: E5a (single versus doubled beginning-of-sequence token), with verdict rules

Written 2026-10-22, BEFORE any E5a cell exists. The user approves E5a (the BOS test that was held back). Apply the rules literally; do not change them after seeing outputs. Everything from earlier briefs still holds: separate output folder, live-engine verification written down before the first cell finishes, nothing deleted or moved in scratch, stop and report on any gate failure, requeue or dirty-GPU event, report only numbers computed from stored per-item records, and until I apply the rules on my laptop report only gate results, per-seed pass@1 and mean completion tokens.

## Why
The campaign pipeline sends every prompt to the engine with two beginning-of-sequence tokens (the prompt text already starts with BOS and add_special_tokens=True adds another; engine prompt tokens = recount + 1). The companion-study runs (P2 cells, and the AWQ cells of the P4 grid) have exactly one token fewer per MATH-500 item, i.e. a single BOS; the P4 BF16 cells match the campaign. The paper compares P2 cells with campaign BF16, so I need to know whether this one token matters.

## 0. Read first (read-only, report back before submitting)
a. Look at how the P2 publication cells obtain a single BOS (resolved_config.json and the P2 runner / vllm model config of job 107150: is it add_special_tokens=False, or a prompt string without BOS, or something else?). Reproduce THAT mechanism in E5a if it can be done with the campaign launcher; tell me exactly what you will change and confirm that it is the only change.
b. Confirm on a CPU that the E5a prompt for item 0 tokenizes to the recount (e.g. 43 tokens for Qwen item 0, not 44) and that the prompt text is otherwise byte-identical to the campaign's.

## 1. E5a
- Model: DeepSeek-R1-Distill-Qwen-7B, BF16, campaign configuration in every other respect (vLLM 0.7.0 eager, T=0.6, top_p=0.95, max tokens 32768, max_num_seqs 256, bfloat16, prefix caching and chunked prefill off).
- Data: MATH-500 (500 items), seeds 42-46 (5 seeds). If time forces it, run seeds 42-44 first (these match the P2 seeds) and the other two afterwards, and say which were run.
- Gate: first cell of the job, engine prompt_tokens == recount on EVERY row (offset 0), prompts otherwise byte-identical to the campaign's, schema and finish_reason as in E2. Launch with R2_EXPECT_OFFSET=0 for this job only; record why. Any other offset fails the gate (exit 3).
- Own output folder (not shared with any other experiment); tag the cells so they cannot collide with E2/E3/X3.

## 2. Verdict rules (fixed now)
Comparison: E5a cells against the campaign BF16 Qwen cells with the same seeds (MATH-500 seeds 42-46), paired item bootstrap (B=10,000, default_rng(0)), 95% intervals, with the estimators of the paper (seed-matched and cross-seed Lambda_b, five-seed BF16 placebo, excess = Lambda_b - Pi_b).
1. "A BOS effect is detected" if ANY of the following holds: (a) the 95% interval of the accuracy difference E5a - BF16 excludes 0; (b) the 95% interval of the unconditional mean-length difference excludes 0; (c) the 95% interval of the cross-seed excess Lambda_b - Pi_b excludes 0.
2. Otherwise "not detected at this sample size". No third category.
3. Descriptive only (no verdict): share of completions token-identical to the campaign run at the same seed, distribution of the first divergent token index, correct-to-wrong flip rate against the BF16-vs-BF16 baseline, and the pass@1 of each seed. Expect almost no token identity (the prompts differ by one token); that is not evidence of an effect on its own.
4. Do not apply these rules yourself in your report; give me the numbers.

## 3. Order and limits
Submit E5a now (nice 0 is fine; it is small, about 2.5 GPU-h for 5 seeds). Do not start anything else. If the job has not started by 2026-10-29, tell me, and tell me the queue reason. If a gate fails, stop and report.

## 4. What to send back
A new archive exports/laptop_export_<date>_v5.tar.gz with the compact per-item JSON of the E5a cells (details[].row, completion_tokens, prompt_tokens, extractive_match, finish_reason), the engine-settings dumps, manifests, cell_status rows, the token-identity JSON against the campaign cells, and the answers to 0a and 0b. Give size, file count, sha256 and the exact scp line (destination C:/Users/manis/Downloads/QuantReasoning_Paper/hpc/) first. No weights, no full completion text.
