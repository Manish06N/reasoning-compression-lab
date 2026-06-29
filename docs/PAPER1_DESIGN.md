Paper 1 title:
Beyond Accuracy: Reliability, Calibration, Seed Variance, and Cost-per-Correct of Quantized Reasoning LLMs

Main question:
When reasoning LLMs are compressed, do they remain accurate, calibrated, stable, fast, memory-efficient, and economically useful?

Core comparison:
BF16 vs FP8 vs AWQ-4 vs GPTQ-4 vs GPTQ-3

Main models:
DeepSeek-R1-Distill-Qwen-1.5B
DeepSeek-R1-Distill-Qwen-7B
DeepSeek-R1-Distill-Llama-8B

Ambitious extension:
Add 14B or Qwen3 thinking model if A100 budget allows.

Main tasks:
MATH-500
GPQA-Diamond
LiveCodeBench subset
GSM8K control

Main metrics:
pass@1
maj@5
Brier
ECE
AURC
seed variance
latency
peak VRAM
cost-per-correct

Main claim:
Single-run accuracy is not enough for deployment decisions.

Publication sufficiency plan:
Use the b01-b09 seed0 HPC grid as the first publishable core if results are clean: Qwen-1.5B, Qwen-7B, Llama-8B across BF16/FP8/AWQ-4/GPTQ-4/GPTQ-3 on MATH-500 plus GSM8K and GPQA checks. Do not expand before scoring this grid. If robustness is required, add seed1/seed2 only for the key Qwen-7B and Llama-8B MATH-500 subset: BF16, FP8, AWQ-4, GPTQ-4.

Full PhD roadmap (thesis spine, tracks, V7 job strategy, GGUF extension):
docs/phd-roadmap/README.md
