# Pass criteria for the live-engine check (written 2026-10-13T05:38:31+05:30, BEFORE any rerun finished; do not change after seeing outputs)

Source of truth for each cell: values read from the live vLLM engine object after load (model_config / scheduler_config / cache_config), written to sidecar.jsonl.engine by the workspace copy of vllm_model.py, plus the logged SamplingParams. Campaign reference: results/reports/runtime_manifest.json (enforce_eager true, gpu_memory_utilization 0.75, max_model_len 32768, max_new_tokens 32768, T 0.6, top_p 0.95, repetition_penalty 1.0, prefix caching off, chunked prefill off, TP 1, vLLM 0.7.0) and vLLM 0.7.0 defaults (max_num_seqs 256, V0 engine).

| Check | Expected | Applies to |
|---|---|---|
| effective dtype | float16 (E1a/E1c self-AWQ, E2) ; bfloat16 (E3, X1, X3, X5, anything BF16 with no dtype override) | per cell |
| effective quantization | None (BF16 cells); 'awq' or 'awq_marlin' (E1a, whichever vLLM resolves, recorded); exactly 'awq' (E1c forced) | per cell |
| enforce_eager | True | all |
| max_model_len | 32768 | all |
| enable_prefix_caching / chunked prefill | both off | all |
| max_num_seqs | 256 (vLLM 0.7.0 default = campaign) ; 64 only for X3 and X5 config B (R2_MAX_NUM_SEQS set) | all |
| SamplingParams | temperature 0.6 (0.8 for E3, 0 for X5), top_p 0.95 (1.0 under greedy: vLLM resets it), max_tokens 32768, repetition_penalty 1.0, seed = cell seed | all |
| first completed cell of a job (MATH-500 n=500) | prompts byte-identical to campaign BF16, engine prompt_tokens == campaign recount (offset 0), raw+compact schema == campaign, finish_reason complete | every loop job |
Failures: config mismatch -> exit 3 (job aborts); engine file missing -> message starts with DUMP_FAILED (harness problem, still exit 3); E2 numerical failure -> exit 4 (FAILED_NUMERIC, only for E2).
verify_cell.py sha256 at time of writing: 428a44d6c231e19b ; run_cells.sh: 6f28bff26430eb02
