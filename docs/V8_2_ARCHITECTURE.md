# V8.2 Architecture Map

PhD Roadmap V8.2 (1 July 2026) — complete codebase layout.

## Python packages (`src/`)

| V8.2 module | Path | Status |
|-------------|------|--------|
| `generation/vllm` | `src/generation/vllm/` | Active (J1 HPC) |
| `generation/sglang` | `src/generation/sglang/` | Pilot stub (J2) |
| `generation/llamacpp` | `src/generation/llamacpp/` | Pilot stub (J3) |
| `evaluation/correctness` | `src/evaluation/correctness/` | Active |
| `evaluation/calibration` | `src/evaluation/calibration/` | Active |
| `evaluation/selective_risk` | `src/evaluation/selective_risk/` | Active |
| `evaluation/statistics` | `src/evaluation/statistics/` | Active |
| `profiling` | `src/profiling/` | Active |
| `schemas` | `src/schemas/` + `schemas/*.json` | Active |

Legacy `src/metrics/`, `src/runners/`, `src/extraction/` remain for HPC backward compatibility.

## Paper protocols

| Paper | Protocol | Scripts |
|-------|----------|---------|
| J1 | `papers/j1/protocol.yaml` | `scripts/j1/*` |
| J2 | `papers/j2/protocol.yaml` | `scripts/j2/*` |
| J3 | `papers/j3/protocol.yaml` | `scripts/j3/*` |

## Config layers

- `configs/models/` — model paths and vLLM args
- `configs/quantization/registry.yaml` — quant method registry
- `configs/serving/` — vLLM, SGLang, llama.cpp pins
- `configs/tasks/` — dataset + prompt template
- `configs/cells/` — experiment cells with `prompt_profile`

## Stable CLIs (do not rename)

```bash
python scripts/run_inference.py --cell-config configs/cells/...
python scripts/score_run.py --input runs/raw/....jsonl
python scripts/hpc/run_hpc_2a100_publication.sh b01_parallel_bf16_anchors
```

## External repos

See `../external_repos/README.md` and run `scripts/record_external_repo_pins.sh`.

## Full plan

[docs/plans/2026-07-01-v82-reengineering.md](plans/2026-07-01-v82-reengineering.md)
