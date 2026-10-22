# Export notes - v4b (built 2026-10-22; read-only on all result trees; written only under exports/)
v4b = v4 plus the 17 cells of Slurm raw job 110901. 110901 is the raw id of array task 108563_0 (sacct JobID 108563_0, JobIDRaw 110901), so these cells belong to job 108563, which is inside the requested range. v4 had wrongly excluded them as "id > 108564".
Scope: P2/P3/P4/P7 jobs COMPLETED between 104321 and 108564 (quantized-reasoning campaign only; kg-*, r2-*, r2x-* excluded). Excluded: 107156 (TIMEOUT), 107157 (CANCELLED) = invalid Llama v1 campaigns; 107059 (FAILED).
tables/cell_inventory.tsv - 140 per-item cells; tables/job_inventory.tsv - one row per completed job (incl. jobs without per-item cells).
compact/*.json - details[].row (1-based), dataset_index, item_id, completion_tokens, prompt_tokens, exact_correct (stored runner `correct`), extractive_match (rescored), finish_reason. No completion text.
cells/<slug>/ - metrics.json (incl. energy fields), resolved_config.json, runtime_kernel_evidence.json, job.json.
Rescoring: stored completion text -> LightEval 0.8.0 (env qrm-official, same as E1-E3) with the campaign metric objects lighteval_custom.tasks.reasoning.latex_gold_metric (MATH-500) / expr_gsm8k_metric (GSM8K). The math_verify pip package is not installed there; LightEval bundles its extraction (latex2sympy2_extended 1.0.6).
Validation (tables/scorer_validation.txt): reproduces stored extractive_match on 500/500 campaign MATH-500 rows and 1319/1319 E1b GSM8K rows. GSM8K gold restored to campaign form (full answer text) by question match; boxed-form gold is not equivalent.
exact_correct is the runner's own scorer, a different measure from extractive_match; do not mix them in one analysis.
Labels in evidence_label_per_docs copy the repo's wording (docs/EXECUTION_STATUS.md, papers/*/RESULTS_STATUS.md); no claim is promoted.
Design documents: design_docs/ (P2, P3, P4, P7 docs + configs, claim ledger, execution status, P4 redesign). P4 design: papers/paper_04_hardware_transfer (EXPERIMENT_MATRIX, HYPOTHESES, RESEARCH_QUESTIONS, CLAIMS_BOUNDARY, RESULTS_STATUS, configs/pilot_decoding_replicates_qwen7b_ready.yaml, job_matrix.p4_decoding_qwen7b_math300_rep3_v1.*) and docs/P4_REDESIGN_2026-10-02.md.

v4c addition (2026-10-22): provenance/GSM8K_item_table.tsv (+README, checks) and provenance/MATH500_item_table.* (copied from the v2 export). Job 110901 (raw id of array task 108563_0, 17 cells) is already included as in v4b. Job 108566 (P2 Llama tokenizer repair) has not run, so there are no Llama v2 cells in this export.
