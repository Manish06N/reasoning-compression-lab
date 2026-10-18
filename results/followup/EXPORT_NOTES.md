# Export notes

Built read-only on 2026-10-18; nothing in the source trees was changed.

- DeepSeek-R1-Distill-Llama-8B-selfawq_gsm8k_n1319_seed42: 1 rows with engine/compact completion_tokens difference outside 0..2
- DeepSeek-R1-Distill-Llama-8B-selfawq_gsm8k_n1319_seed43: 3 rows with engine/compact completion_tokens difference outside 0..2
- DeepSeek-R1-Distill-Llama-8B-selfawq_gsm8k_n1319_seed44: 3 rows with engine/compact completion_tokens difference outside 0..2
- DeepSeek-R1-Distill-Llama-8B_T0.8_math500_n500_seed42: 1 rows with engine/compact completion_tokens difference outside 0..2
- DeepSeek-R1-Distill-Llama-8B_T0.8_math500_n500_seed44: 2 rows with engine/compact completion_tokens difference outside 0..2
- DeepSeek-R1-Distill-Llama-8B_T0.8_math500_n500_seed46: 1 rows with engine/compact completion_tokens difference outside 0..2
- DeepSeek-R1-Distill-Llama-8B_fp16_math500_n500_seed42: 3 rows with engine/compact completion_tokens difference outside 0..2
- DeepSeek-R1-Distill-Llama-8B_fp16_math500_n500_seed43: 1 rows with engine/compact completion_tokens difference outside 0..2
- DeepSeek-R1-Distill-Llama-8B_fp16_math500_n500_seed46: 1 rows with engine/compact completion_tokens difference outside 0..2
- DeepSeek-R1-Distill-Llama-8B_heldout_math500_n500_seed48: 2 rows with engine/compact completion_tokens difference outside 0..2
- DeepSeek-R1-Distill-Llama-8B_heldout_math500_n500_seed50: 1 rows with engine/compact completion_tokens difference outside 0..2
- DeepSeek-R1-Distill-Llama-8B_heldout_math500_n500_seed51: 1 rows with engine/compact completion_tokens difference outside 0..2
- DeepSeek-R1-Distill-Qwen-7B_T0.8_math500_n500_seed42: 1 rows with engine/compact completion_tokens difference outside 0..2
- DeepSeek-R1-Distill-Qwen-7B_T0.8_math500_n500_seed43: 1 rows with engine/compact completion_tokens difference outside 0..2
- DeepSeek-R1-Distill-Qwen-7B_T0.8_math500_n500_seed44: 3 rows with engine/compact completion_tokens difference outside 0..2
- DeepSeek-R1-Distill-Qwen-7B_T0.8_math500_n500_seed45: 2 rows with engine/compact completion_tokens difference outside 0..2
- DeepSeek-R1-Distill-Qwen-7B_fp16_math500_n500_seed42: 1 rows with engine/compact completion_tokens difference outside 0..2
- DeepSeek-R1-Distill-Qwen-7B_fp16_math500_n500_seed44: 2 rows with engine/compact completion_tokens difference outside 0..2
- DeepSeek-R1-Distill-Qwen-7B_fp16_math500_n500_seed45: 1 rows with engine/compact completion_tokens difference outside 0..2
- DeepSeek-R1-Distill-Qwen-7B_heldout_math500_n500_seed51: 1 rows with engine/compact completion_tokens difference outside 0..2
- E1 cell dir collision: DeepSeek-R1-Distill-Llama-8B-selfawq_math500_n500_seed42: raw output, sidecar (finish_reason), MANIFEST, VERIFIED, VERIFY.txt and engine dump were overwritten by the E1b GSM8K run in the same inference dir; compact JSON and cell_status row remain
- E1 cell dir collision: DeepSeek-R1-Distill-Llama-8B-selfawq_math500_n500_seed43: raw output, sidecar (finish_reason), MANIFEST, VERIFIED, VERIFY.txt and engine dump were overwritten by the E1b GSM8K run in the same inference dir; compact JSON and cell_status row remain
- E1 cell dir collision: DeepSeek-R1-Distill-Llama-8B-selfawq_math500_n500_seed44: raw output, sidecar (finish_reason), MANIFEST, VERIFIED, VERIFY.txt and engine dump were overwritten by the E1b GSM8K run in the same inference dir; compact JSON and cell_status row remain
- MISSING (not produced on HPC): revision2/REPORT_diagnostics.json
- MISSING (not produced on HPC): revision2/REPORT_diagnostics.md
- MISSING (not produced on HPC): revision2/RUNS.csv
- MISSING: no MANIFEST.md exists anywhere; per-cell MANIFEST.json files are included instead
- X1 analysis output (x1_analysis.py) and the Qwen E2 seed-42 repeat-agreement report were never produced on the HPC; not generated here.
- p2_dtype compact files use exact-match `exact_correct` (scoring mode exact_only_rescore_required), NOT math-verify `extractive_match`; do not mix with revision2 compact files without rescoring.
- p2_dtype settings are the REQUESTED model_config plus detected kernel; max_num_seqs was not dumped for those jobs.
