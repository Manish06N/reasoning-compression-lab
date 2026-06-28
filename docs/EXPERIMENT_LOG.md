# Experiment Log

Chronological record of runs. One section per experiment cell.

---

## Template

```text
Date:
Level:        A | B | C
Model:
Quant config:
Task:
Seed(s):
Hardware:
Status:       planned | running | done | failed
Raw path:
Scored path:
Notes:
Key numbers:
```

---

## 2026-06-26 — Project setup (MacBook)

```text
Date:         2026-06-26
Level:        setup
Status:       done
Notes:        Repo structure created. Design doc and paper skeleton written.
              Next: HPC env + Qwen-7B download + BF16 MATH-500 seed 0.
```

---

## 2026-06-28 — Windows 5080 publication main grid (running)

```text
Date:         2026-06-28
Level:        B/C main (5080 primary)
Hardware:     RTX 5080 16GB (WSL2), vLLM 0.23, torch 2.11+cu128
Status:       running (publication main grid)
Archive:      outputs-win5080-main-2026-06-28/
Protocol:     repro_qrm.yaml — temp 0.6, top_p 0.95, max_tokens 32768, seed 0, batch_size=1
Sample sizes: MATH-500 n=500, GSM8K n=1319
Policy:       5080 primary; HPC only for >16GB VRAM (BF16 7B/8B)
Skipped:      BF16 7B/8B (VRAM), GPQA (gated)
Notes:        Supersedes pilot archive (outputs-win5080-pilot-*). Pilot not for paper.
Command:      bash scripts/local/start_5080_main.sh
```

---

## 2026-06-28 — Windows 5080 pilot pipeline (superseded)

```text
Date:         2026-06-28
Level:        pilot (5080 debug)
Hardware:     RTX 5080 16GB (WSL2), vLLM 0.23, torch 2.11+cu128
Status:       superseded — do not cite in paper
Archive:      outputs-win5080-pilot-2026-06-28/
Protocol:     n=50, pilot_5080.yaml (max_tokens 8192), batched inference
Cells:        14 (smoke + 13 inference); BF16 7B/8B + GPQA skipped
Backup:       _backup/latest/ + snapshots every 3 cells
Resume:       bash scripts/local/resume_5080_pilot.sh
Notes:        Full grid aborted; pilot supersedes. 12 models on disk.
```

---

## 2026-06-28 — Windows 5080 setup + pilot protocol

```text
Date:         2026-06-28
Level:        setup + pilot infra
Hardware:     RTX 5080 16GB (WSL2), vLLM 0.23, torch 2.11+cu128
Status:       done (infra); pilot grid ready to run
Notes:        All 12 model checkpoints downloaded (~62 GB).
              Phase 0 smoke: Qwen-1.5B BF16 OK; Qwen-7B BF16 OOM (HPC only).
              Full grid started then superseded by pilot mode (batched n=50).
              Pilot archive: outputs-win5080-pilot-2026-06-28/
              Full archive:  outputs-win5080-2026-06-28/
Key numbers:  ~1-4k completion tokens/q on 1.5B; ~15 tok/s; batch 4/2/1 by model size
```

---

## Level A — Reproduction gate (pending)

```text
Date:
Level:        A
Model:        DeepSeek-R1-Distill-Qwen-7B
Quant config: BF16
Task:         MATH-500
Seed(s):      0
Hardware:     A100
Status:       planned
Notes:        First truth test. Must pass before GPTQ-4 or pilot grid.
```

---

## Level A — GPTQ-4 reproduction (blocked until prep)

```text
Date:
Level:        A
Model:        DeepSeek-R1-Distill-Qwen-7B
Quant config: GPTQ-4
Task:         MATH-500
Seed(s):      0
Hardware:     A100
Status:       blocked
Notes:        Requires BF16 Level A done first.
              Then: download/quantize GPTQ-4 → 06_verify_gptq4_model.sh → inference.
              See docs/GPTQ4_PREP.md.
```
