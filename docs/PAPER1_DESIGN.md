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
