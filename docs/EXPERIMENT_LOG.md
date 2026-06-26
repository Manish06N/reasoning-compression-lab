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

## Level A — Reproduction gate (pending)

```text
Date:
Level:        A
Model:        DeepSeek-R1-Distill-Qwen-7B
Quant config: BF16
Task:         MATH-500
Seed(s):      0
Hardware:     A100 (PARAM Rudra, IIT Patna)
Status:       running
Notes:        HPC setup complete through Gate 2b. Smoke test submitted (check slurm logs).
              Next: Gate 3 pass → 10-question debug → full MATH-500.
```

---

## 2026-06-26 — HPC setup (PARAM Rudra)

```text
Date:         2026-06-26
Level:        setup
Hardware:     PARAM Rudra — /scratch/manishn_iitp/reasoning-compression-lab
Status:       in progress
Gate 1:       PASSED — GPU + torch + vLLM
Gate 2:       DONE — Qwen-7B downloaded
Gate 2b:      DONE — MATH-500 validated (500 examples)
Gate 3:       SUBMITTED — sbatch slurm/smoke_test.slurm
Notes:        HF auth + hf_cache on scratch. HPC commit synced via GitHub from MacBook.
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
