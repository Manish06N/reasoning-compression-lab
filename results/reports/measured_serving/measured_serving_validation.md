# Measured Serving Benchmark Scientific Validation Audit

**Audit Timestamp:** 2026-08-16  
**Hardware Platform:** NVIDIA A100-PCIE-80GB  
**Total Executed Runs:** 48 task-realistic + 8 microbenchmark runs  

## Integrity Checks
- **All 8 Configurations Completed:** YES (Qwen-7B / Llama-8B × BF16, FP8, AWQ-4, GPTQ-4)
- **Both Serving Conditions Measured:** YES (Condition A: C=1, Condition B: C=8)
- **Repetitions per Condition:** 3 wall-clock repeats with shared sampling seed 20260816 (identical token counts)
- **Input Prompt Subset:** Frozen 100 MATH-500 prompts stratified across Levels 1–5
- **Raw JSON completeness:** 48 task-realistic + 8 microbenchmark files present; tok/s and GPU-sec/query recompute from elapsed/tokens
- **OOM / SLURM errors:** no OOM strings in raw JSON; SLURM logs are not in git, so 0-failure is not independently proven from this artifact
- **Node mix:** Llama AWQ-4 and Llama GPTQ-4 have records from more than one hostname (cache reuse / re-execution), so ``0 restarts'' is not verified
- **Protocol notes:** Condition B is a 100-prompt `llm.generate` (continuous batching; `max_num_seqs` not pinned to 8). Condition A uses the first 20 prompts of the frozen list (level counts 5/7/3/3/2). Repetitions share sampling seed 20260816 (identical token counts). Peak VRAM is allocated bytes after `gpu_memory_utilization=0.75`, not isolated weight footprint.
