# Ambitious Extension — Stack Transfer (GGUF, KV-Cache, Backends)

[← Index](README.md) · Paper 1 minimum grid: [04_PAPER1_FULL_SPEC.md](04_PAPER1_FULL_SPEC.md)

---

## Big picture

| Layer | Current run (minimum) | Ambitious extension |
|-------|----------------------|---------------------|
| **Question** | Under **vLLM**, how do BF16/FP8/AWQ/GPTQ affect accuracy, calibration, seed variance, VRAM, latency, cost? | Do conclusions **hold** on **llama.cpp/GGUF** and other backends? |
| Model weights | BF16, FP8, AWQ, GPTQ-4, GPTQ-3 | + GGUF Q8/Q6/Q4/Q3 |
| Runtime | vLLM (A100) | + llama.cpp (edge) + SGLang (datacenter compare) |
| Hardware | HPC A100 | A100 datacenter + local edge where stable |
| Memory focus | Weight memory | Weight + **KV-cache** memory |
| Metrics | Accuracy, calibration, cost | Same + **stack transfer** + energy (if clean) |

**Thesis alignment:** The spine is **deployment**, not quantization alone. GGUF/llama.cpp is a **deployment format**, not just another quant method.

**Rule:** Finish and score the **current vLLM grid (b01–b09)** before starting ambitious extensions. Do not add formats in parallel mid-run.

---

## Priority 1 — GGUF formats (most important)

GGUF packages weights for **llama.cpp** — the main local/edge deployment path.

### Recommended GGUF variants

| GGUF format | Role |
|-------------|------|
| **Q8_0** | Near-reference local baseline vs BF16/FP8 |
| **Q6_K / Q6_K_M** | Strong quality/size balance |
| **Q5_K_M** | Middle ground |
| **Q4_K_M** | Main practical 4-bit local baseline |
| **Q3_K_M** | Aggressive; compare with GPTQ-3 |

### First GGUF grid (do not add every model at once)

| Model | GGUF formats | Why |
|-------|--------------|-----|
| Qwen-7B | Q8_0, Q6_K, Q4_K_M, Q3_K_M | Main local model |
| Llama-8B | Q8_0, Q6_K, Q4_K_M | Architecture comparison |
| Qwen-1.5B | Q8_0, Q4_K_M | Sanity / control |

**Tasks:** MATH-500 + GSM8K (7B/8B); MATH-500 only (1.5B).  
**Seeds:** 3 first, then 5 if trends are clear.

---

## Priority 2 — KV-cache quantization

Weight quantization ≠ KV-cache quantization. Long reasoning traces inflate KV memory.

| KV option | Context |
|-----------|---------|
| FP16/BF16 KV | Reference |
| FP8 KV | vLLM / A100 deployment |
| llama.cpp q8_0 KV | Local baseline |
| llama.cpp q4_0 KV | Aggressive local |

**Study design:** Same model + task, vary KV mode holding weight format fixed.

Example cells:

| Model | Weights | KV modes |
|-------|---------|----------|
| Qwen-7B | vLLM FP8 or GPTQ-4 | default vs FP8 KV |
| Qwen-7B | GGUF Q4_K_M | default vs q8/q4 KV (if stable) |

Novelty axis: **deployment measurement** + **stack sensitivity** (roadmap differentiators).

---

## Priority 3 — Backend comparison (SGLang)

Same model + quant + task, different **serving stack**:

| Config | Backend |
|--------|---------|
| Qwen-7B GPTQ-4 | vLLM |
| Qwen-7B AWQ-4 | vLLM |
| Qwen-7B GGUF Q4_K_M | llama.cpp |
| Qwen-7B FP8 | vLLM / SGLang (if supported) |

**Question:** Does the **ranking** of quant formats change when the backend changes?

Publishable claim: *Deployment decisions depend on both compression format and serving stack.*

---

## Priority 4 — Qwen-14B (only after core stable)

| Model | Formats | Hardware |
|-------|---------|----------|
| Qwen-14B BF16 | Reference | A100 |
| Qwen-14B FP8 | Serving | A100 |
| Qwen-14B AWQ-4 / GPTQ-4 | Quant | A100 |
| Qwen-14B GGUF Q4_K_M | Edge exploratory | Local only if VRAM allows |

Do **not** add 32B/70B unless supervisor explicitly requests — timeline risk.

---

## Priority 5 — Energy / joules-per-correct

Secondary unless measurement is clean (repeated samples, NVML caveats, wall-power validation).

| Metric | Meaning |
|--------|---------|
| Joules per answer | Energy per generation |
| Joules per **correct** answer | Energy efficiency for useful outputs |
| Tokens/sec/watt | Throughput efficiency |
| cost-per-correct | Primary industry metric (already in P1) |

Strong fit for **Paper 3** or ambitious P1 appendix — do not block J1 submission on noisy energy.

---

## What NOT to add immediately

| Format / method | Recommendation |
|-----------------|----------------|
| SmoothQuant | Skip unless pivot |
| bitsandbytes NF4 | Training-oriented; less central for serving compare |
| ONNX | Skip unless production deployment angle |
| TensorRT-LLM | Optional later; engineering-heavy |
| EXL2 / ExLlamaV2 | Enthusiast format; less journal-clean than GGUF/vLLM |
| Every GGUF variant | Pick 3–4 useful ones only |

---

## Recommended ambitious extension packages

### Extension A — GGUF / llama.cpp stack-transfer

| Model | Formats | Tasks | Seeds |
|-------|---------|-------|-------|
| Qwen-7B | Q8_0, Q6_K, Q4_K_M, Q3_K_M | MATH-500, GSM8K | 3 → 5 |
| Llama-8B | Q8_0, Q6_K, Q4_K_M | MATH-500, LCB subset | 3 → 5 |
| Qwen-1.5B | Q8_0, Q4_K_M | MATH-500 | 3 |

### Extension B — KV-cache compression

See Priority 2 table above.

### Extension C — Backend comparison

| Same model | Compare |
|------------|---------|
| Qwen-7B 4-bit | vLLM GPTQ-4 vs AWQ-4 vs llama.cpp GGUF Q4_K_M |
| Llama-8B 4-bit | vLLM GPTQ-4 vs llama.cpp GGUF Q4_K_M |

---

## Simple mental model

```
Current:   vLLM × {BF16, FP8, AWQ, GPTQ} × tasks × seeds
Ambitious: + llama.cpp × {GGUF Q8, Q6, Q4, Q3}
           + KV-cache modes
           + SGLang (optional)
           + Qwen-14B (optional)
```

**Research question for extension:**

> If GPTQ-4 looks best in vLLM, does GGUF Q4_K_M also look best on llama.cpp — or does the conclusion **change**?

That is a strong, publishable stack-transfer claim aligned with the thesis spine.

---

## Final recommended order (after current grid completes)

1. GGUF **Q8_0**  
2. GGUF **Q4_K_M**  
3. GGUF **Q6_K / Q6_K_M**  
4. GGUF **Q3_K_M**  
5. KV-cache quantization  
6. SGLang backend comparison  
7. Qwen-14B (if A100 budget allows)  

**Do not add everything at once.** First extension: **Qwen-7B GGUF Q8 + Q4 + Q3** — cleanest next step after vLLM grid is scored.

---

## Repo integration (when implemented)

Suggested future additions (not in minimum grid yet):

- `configs/models/*_gguf.json`  
- `src/runners/llamacpp_runner.py` (or equivalent)  
- `configs/machine_split/hpc_blocks/b10_stack_transfer_gguf.sh`  
- Document HF GGUF sources in [docs/MODEL_ROSTER.md](../MODEL_ROSTER.md)  

Track progress in [progress.md](../../progress.md) when extension work starts.
