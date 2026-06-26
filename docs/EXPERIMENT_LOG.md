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
