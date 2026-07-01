# RTX 5080 (16 GB) — Paper 1 Execution Plan

**Machine:** Windows 11 + WSL2 + NVIDIA GeForce RTX 5080 (Blackwell sm_120)  
**Stack:** vLLM 0.23 + torch 2.11+cu128 (local); `enforce_eager=true`, `kv_cache_dtype=fp8` on quants  
**Output archive (publication):** `outputs-win5080-main-2026-06-28/`  
**Pilot archive (debug only):** `outputs-win5080-pilot-2026-06-28/`

---

## Machine policy (journal)

| Where | When |
|-------|------|
| **RTX 5080** | **Only** cells expected to finish in **≤24 h each** — Qwen-1.5B × 4 quants × MATH-500 |
| **HPC 2× A100** | **Everything else** — all 7B/8B quants, BF16 anchors, GSM8K, GPQA (≤48 h SLURM blocks) |

See [HPC_2A100_PLAN.md](HPC_2A100_PLAN.md) for the full block grid.

---

## Are your reports enough?

**Yes for planning and for the full 5080 grid.** They correctly describe VRAM math, quant trade-offs, and which checkpoints to use. One correction from live runs on your 5080:

- **Qwen-7B BF16 is not sustainable locally** with vLLM 0.23 — weights ~14.3 GiB leave **no KV cache** on 16 GB (confirmed OOM). Treat BF16 7B as **HPC-only** for full MATH-500, not “comfortably runnable” on 5080.

---

## Best models for Paper 1 (canonical picks)

Use **one HF ID per quant slot**. Prefer **DeepSeek official** for BF16 and **RedHatAI / jakiAJK** for vLLM-documented quants.

| Priority | Role | Model | HF ID | 5080 |
|----------|------|-------|-------|------|
| **1** | Pipeline verifier | Qwen-1.5B BF16 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | ✅ Always |
| **2** | Paper anchor (quant) | Qwen-7B FP8 | `RedHatAI/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic` | ✅ **Best default** |
| **3** | Paper anchor (4-bit) | Qwen-7B GPTQ-4 | `RedHatAI/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16` | ✅ Level A repro |
| **4** | 4-bit alternate | Qwen-7B AWQ-4 | `jakiAJK/DeepSeek-R1-Distill-Qwen-7B_AWQ` | ✅ |
| **5** | Architecture compare | Llama-8B FP8 | `RedHatAI/DeepSeek-R1-Distill-Llama-8B-FP8-dynamic` | ✅ Quant only |
| **6** | Lower bound | Qwen-1.5B FP8/GPTQ-4 | RedHatAI 1.5B quants | ✅ Calibration |
| **7** | Experimental | Qwen-7B GPTQ-3 | `irish-quant/deepseek-ai-DeepSeek-R1-Distill-Qwen-7B-3bit` | ⚠️ Run last |

**HPC only (not 5080 full runs):**

| Model | Why |
|-------|-----|
| Qwen-7B BF16 | Weights alone exceed 16 GB headroom for KV cache |
| Llama-8B BF16 | ~16 GB weights — OOM on load |
| GPQA-Diamond | Gated dataset |
| 14B / 32B / 70B | Out of scope for 16 GB |

**Reject from reports:** GGUF/llama.cpp, uncensored forks, `ModelCloud/...-vortex-v2` unless verified, mixed GPTQ sources within one label.

---

## Full 5080 roster (4 inference cells + smoke)

Queue: `configs/machine_split/5080_cells.sh` — sequential, one model at a time:

| # | Phase | Cell | Model × Quant | Task |
|---|-------|------|---------------|------|
| 0 | Smoke | `smoke_qwen15b_bf16` | Qwen-1.5B BF16 | 1 question |
| 1 | C | `level_c_qwen15b_bf16_math500_seed0` | Qwen-1.5B BF16 | MATH-500 |
| 2 | C | `level_c_qwen15b_fp8_math500_seed0` | Qwen-1.5B FP8 | MATH-500 |
| 3 | C | `level_c_qwen15b_awq4_math500_seed0` | Qwen-1.5B AWQ-4 | MATH-500 |
| 4 | C | `level_c_qwen15b_gptq4_math500_seed0` | Qwen-1.5B GPTQ-4 | MATH-500 |

**7B/8B cells (rows 5–13 in older plans) → HPC blocks b01–b06**, not 5080.

Decoding (full repro): `configs/decoding/repro_qrm.yaml` — temp 0.6, top_p 0.95, max_tokens 32768, seed 0.

---

## Publication main mode (default — journal results)

Run the **4-cell 1.5B grid** at QRM reproduction standard (~4 days total):

| Setting | Main (publication) | Pilot (optional debug) |
|---------|-------------------|------------------------|
| Command | `bash scripts/local/start_5080_main.sh` | `bash scripts/local/run_5080_pilot.sh` |
| Archive | `outputs-win5080-main-YYYY-MM-DD/` | `outputs-win5080-pilot-YYYY-MM-DD/` |
| Sample size | 500 (MATH) / 1319 (GSM8K) | 50 |
| Decoding | `configs/decoding/repro_qrm.yaml` | `configs/decoding/pilot_5080.yaml` |
| max_tokens | 32768 | 8192 |
| vLLM batching | **1** (sequential) | 4 / 2 / 1 |
| Reproducibility | `VLLM_BATCH_INVARIANT=1` | not required |
| Paper tables | **Yes** | **No** |

Resume:

```bash
bash scripts/local/resume_5080_main.sh --skip-download
bash scripts/local/clean_5080_run.sh
```

---

## Pilot mode (optional — pipeline debug only)

Run the **full 14-cell grid at n=50** in hours instead of weeks:

| Setting | Pilot | Full repro |
|---------|-------|------------|
| Command | `bash scripts/local/run_5080_pilot.sh` | `bash scripts/local/run_all_5080_phases.sh` |
| Archive | `outputs-win5080-pilot-YYYY-MM-DD/` | `outputs-win5080-YYYY-MM-DD/` |
| Sample size | 50 / task | 500 (MATH) or 1319 (GSM8K) |
| Decoding | `configs/decoding/pilot_5080.yaml` | `configs/decoding/repro_qrm.yaml` |
| max_tokens | 8192 | 32768 |
| max_model_len | 8192 | 32768 (model default) |
| vLLM batching | 4 (1.5B), 2 (7B/8B) | 1 (legacy sequential) |

Pilot keeps **temperature 0.6 / top_p 0.95** — only length and sample size change.  
Use pilot to rank quants; run **full MATH-500 on anchors** (7B FP8, 7B GPTQ-4) on HPC or overnight.

Resume / skip-download:

```bash
bash scripts/local/run_5080_pilot.sh --skip-download
bash scripts/local/run_all_5080_phases.sh --pilot --limit 25   # custom subset
```

---

## Commands (Windows / WSL)

```bash
wsl -d Ubuntu-22.04
cd "/mnt/g/ALL MY Projects/2026/03-paper1-experiments"
source scripts/local/env.sh

# 1) Download all 5080-fit weights (~30–50 GB total)
bash scripts/local/download_models.sh 5080

# 2) Pilot grid (recommended first — ~hours)
bash scripts/local/run_5080_pilot.sh

# 3) Debug pass (10 problems per task)
bash scripts/local/run_all_5080_phases.sh --limit 10

# 4) Full paper cells (days–weeks on 5080; HPC for BF16 anchors)
bash scripts/local/run_all_5080_phases.sh
```

**Monitor:** `outputs-win5080-2026-06-28/logs/master.log` and `manifest.json`

---

## VRAM settings (5080)

| Setting | Value |
|---------|--------|
| `tensor_parallel_size` | 1 |
| `enforce_eager` | true |
| `kv_cache_dtype` | fp8 (7B/8B quants) |
| `gpu_memory_utilization` | 0.90–0.95 |
| `VLLM_USE_FLASHINFER_SAMPLER` | 0 (Blackwell) |

---

## Runtime expectations

| Task | Problems | Pilot (n=50, batch) | Full repro |
|------|----------|---------------------|------------|
| MATH-500 | 500 | ~30–90 min / cell | hours–days / cell |
| GSM8K | 1,319 | ~30–60 min (n=50) | hours / cell |

Checkpoints every 10 rows — safe to stop/resume after reboot.  
Pilot and full archives are **separate folders** — do not merge without labeling.

### Backup and resume (power-cut safe)

- **Atomic JSONL writes** — checkpoint goes to `.jsonl.tmp` then rename (no half-written files).
- **`_backup/latest/`** — auto-synced after each inference checkpoint and each completed cell.
- **`_backup/snapshots/`** — full timestamped copy every 3 cells.
- **`checkpoints/{cell_id}.json`** + **`state.json`** — orchestrator progress.
- **Corrupt JSONL** — auto-restore from `_backup/latest/raw/` if detected.
- **Resume:** `bash scripts/local/resume_5080_pilot.sh` (same as pilot; skips finished cells).
