# Repository map (V8.2)

Quick navigation for `reasoning-compression-lab`. For the full high-level overview (architecture, papers, gates, status), see **[CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md)**. Read [docs/README.md](README.md) for the doc index.

---

## Start here

| I want to… | Open |
|------------|------|
| Understand the whole codebase | [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md) |
| Run on HPC | [BEGINNER_HPC_GUIDE.md](BEGINNER_HPC_GUIDE.md) |
| See current status | [PROGRESS.md](PROGRESS.md) |
| Avoid known traps | [KNOWN_ISSUES.md](KNOWN_ISSUES.md) |
| Decide which models to add | [MODEL_SCOPE_DECISION.md](MODEL_SCOPE_DECISION.md) |
| Find HF IDs and env vars | [MODEL_ROSTER.md](MODEL_ROSTER.md) |
| Full change history | [../CHANGELOG.md](../CHANGELOG.md) |

---

## Directory tree

```
reasoning-compression-lab/
├── configs/
│   ├── cells/              # Experiment cells (model × task × quant × seed)
│   ├── models/             # vLLM model paths and args
│   ├── tasks/              # Dataset IDs and prompt template refs
│   ├── decoding/           # Temperature, max_tokens, repetition_penalty
│   ├── quantization/       # Quant method registry (V8.2)
│   ├── serving/            # vLLM / SGLang / llama.cpp pins (V8.2)
│   ├── machine_split/      # HPC block → cell wiring (b01–b09)
│   └── baselines/          # QRM literature sanity bands
├── prompts/
│   ├── math500.txt         # Sober (main grid)
│   ├── qrm_math500.txt     # QRM reproduction
│   └── gpqa_diamond.txt
├── papers/
│   ├── j1/protocol.yaml    # Paper 1 frozen endpoints + gates
│   ├── j2/protocol.yaml    # Paper 2 method-selection pilot
│   └── j3/                 # Indic deployment + language_matrix.yaml
├── schemas/                # JSON Schema for raw rows + summaries
├── src/
│   ├── generation/         # vLLM (active), SGLang/llama.cpp (stubs)
│   ├── evaluation/         # Correctness, calibration, selective risk, stats
│   ├── runners/            # Config, vLLM, checkpoints (legacy path, still used)
│   ├── metrics/            # Scoring, cost, Pareto (legacy re-exports)
│   ├── extraction/         # Answer parsers
│   ├── profiling/          # GPU stats, latency
│   └── schemas/            # Provenance + validation
├── scripts/
│   ├── run_inference.py    # ★ Main inference CLI
│   ├── score_run.py        # ★ Main scoring CLI
│   ├── j1/                 # Compare configs, audit, seed aggregate
│   ├── j2/                 # Method pilot manifest
│   ├── j3/                 # Indic preflight, local transfer
│   └── hpc/                # PARAM Rudra orchestration
├── slurm/                  # SLURM job files
├── tests/                  # 31 unit tests (2026-07-01)
├── dashboards/             # HTML dashboard output
└── docs/                   # All documentation
```

---

## Pipeline flow

```
configs/cells/*.json
    → scripts/run_inference.py
    → raw/*.jsonl
    → scripts/score_run.py
    → scored/*.jsonl + results/*_summary.json
    → build_paper_tables.py / build_dashboard.py
```

Publication archives use `outputs-hpc-2a100-main-YYYY-MM-DD/` instead of `runs/`.

---

## External reference code (read-only)

Sibling folder: `../external_repos/` — see [../external_repos/README.md](../../external_repos/README.md).

Never develop experiments inside external repos.

---

## Test suite

```bash
python -m pytest tests/ -q          # expect 31 passed
python scripts/verify_decoding_params.py   # VERIFY OK
```

Test files: `test_config_and_tasks`, `test_math_extractor`, `test_sampling_params`, `test_gpu_stats`, `test_external_repos_integration`, `test_v82_*`.
