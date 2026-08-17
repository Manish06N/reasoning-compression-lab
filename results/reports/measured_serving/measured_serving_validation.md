# Measured Serving Benchmark Scientific Validation Audit

**Audit Timestamp:** 2026-08-16  
**Hardware Platform:** NVIDIA A100-PCIE-80GB  
**Total Executed Runs:** 48 task-realistic + 8 microbenchmark runs  

## Integrity Checks
- **All 8 Configurations Completed:** YES (Qwen-7B / Llama-8B × BF16, FP8, AWQ-4, GPTQ-4)
- **Both Serving Conditions Measured:** YES (Condition A: C=1, Condition B: C=8)
- **Repetitions per Condition:** Exactly 3 independent runs ($R=3$)
- **Input Prompt Subset:** Frozen 100 MATH-500 prompts stratified across Levels 1–5
- **Out-of-Memory (OOM) Events:** 0
- **Job Failures / Restarts:** 0
- **Protocol Deviations:** 0 (all configs executed under identical frozen parameters)
