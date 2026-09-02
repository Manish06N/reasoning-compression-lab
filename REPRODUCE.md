# Reproduce Paper 1 tables (CPU) vs replay the GPU campaign

This repository is a **stack-pinned measurement study**. Frozen numbers come from the completed 88-run GPU campaign. Do not launch new GPU jobs to “improve” results.

The contribution is: under a fixed serving stack, public quantized reasoning checkpoints can change rank depending on what is measured. It is **not** a claim that one quantization method is best.

---

## Two different things

| What | Reproducible on a laptop? |
|------|---------------------------|
| **Tables and statistics** from compact JSON | **Yes.** Stdlib Python 3.11. No GPU. |
| **Full 88-run GPU campaign** and confirmation serving | **No**, not for a typical reviewer. Needs A100-80GB, vLLM 0.7.0, checkpoint weights, dataset SHAs. Full CoT traces are **not** released. |

Do not overclaim: **tables are reproducible; the GPU campaign is inspectable, not expected to be rerun.**

The released artifact enables verification of reported analyses; reproducing the complete GPU campaign requires equivalent hardware and checkpoint availability.

---

## CPU environment (table checks)

```bash
python3 --version   # 3.11 used in CI; 3.10+ is enough for stdlib scripts
```

No `pip install` is required for:

- `revision_reanalysis.py --check`
- `emit_major_revision_tables.py --check`
- `measured_serving_confirmation_analysis.py --check`
- `benchmark_serving_confirmation.py --check`
- `item_level_descriptive_analysis.py --check`
- `validate_runtime_manifest.py --check`
- `check_manuscript_numbers.py --check`
- `check_tex_tables.py --check`

`modal_agreement_analysis.py --check-artifact` is also stdlib. It may print `ERROR importing lighteval` on a MacBook without `qrm-official`; that warning is ignorable if the script then prints `OK: compact artifact SHA256...`.

Optional local tests (not needed to recompute tables):

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q
```

---

## GPU campaign environment (replay only)

Do **not** use `configs/serving/vllm.yaml` (it still says vLLM 0.8.5) or `configs/legacy_models/*.json`.

| Item | Value |
|------|--------|
| Lockfile | `requirements-qrm-paper-vllm070.lock` |
| Installer | `scripts/hpc/qrm_parity/install_official_qrm_env.sh` |
| Python | 3.11 (`qrm-official`) |
| PyTorch | 2.5.1 |
| CUDA toolkit | 12.4 |
| vLLM | **0.7.0** eager |
| LightEval | **0.8.0** (QRM third-party; campaign evaluator) |
| GPU | NVIDIA A100-PCIE-80GB |
| NVIDIA driver | **UNRECORDED** — driver version was unavailable in the archived environment; per-job `nvidia-smi` logs were not retained |
| FP8 execution | Marlin **W8A16**, not native W8A8 |
| Manifest | `results/reports/runtime_manifest.json` |

---

## Commands to regenerate / verify tables

From the repository root. `--check` **does not rewrite** frozen campaign JSON.

```bash
python3 scripts/analysis/revision_reanalysis.py --check
# expected: OK: recomputed report matches .../revision_reanalysis_report.json

python3 scripts/analysis/emit_major_revision_tables.py --check
# expected: OK: generated tables match .../major_revision_tables.md and .../major_revision_validation.md

python3 scripts/analysis/measured_serving_confirmation_analysis.py --check
# expected: OK: recomputed confirmation report matches ...measured_serving_confirmation_report.json (...)

python3 scripts/hpc/qrm_parity/benchmark_serving_confirmation.py --check
# expected: OK: confirmation raw artifacts present (60 required files)

python3 scripts/analysis/modal_agreement_analysis.py --check-artifact
# expected: OK: compact artifact SHA256, 20,000/4,000 structure, T5 accounting, and report internals match.

python3 scripts/analysis/item_level_descriptive_analysis.py --check
# expected: OK: item-level descriptive report matches .../item_level_descriptive_report.json

python3 scripts/analysis/validate_runtime_manifest.py --check
# expected: OK: runtime_manifest.json matches campaign counts, LightEval 0.8.0, vLLM 0.7.0, task SHAs, and configs/publication/INDEX.json.

python3 scripts/analysis/check_manuscript_numbers.py --check
# expected: OK: 15 frozen manuscript needles present in paper/main.tex

python3 scripts/check_tex_tables.py --check
# expected: No manuscript drift detected.
```

To **rewrite** analysis markdown from compact JSON (still no GPU; do this only if you intend to refresh frozen files):

```bash
python3 scripts/analysis/revision_reanalysis.py
python3 scripts/analysis/emit_major_revision_tables.py
python3 scripts/analysis/measured_serving_confirmation_analysis.py
python3 scripts/analysis/item_level_descriptive_analysis.py
```

LaTeX tables in `paper/main.tex` are transcribed from `results/reports/major_revision_tables.md`. `check_manuscript_numbers.py` guards a subset of those frozen strings. It does not prove every TeX cell equals every markdown cell.

---

## Frozen sources of truth

| Artifact | Role |
|----------|------|
| `results/{math500,gsm8k,gpqa}/*.json` | Compact per-cell records (40+24+24 = 88 files) |
| `results/reports/revision_reanalysis_report.json` | Canonical pass@1 / length / Holm |
| `results/reports/major_revision_tables.md` | Frozen paper tables |
| `results/reports/runtime_manifest.json` | Effective launch stack + dataset SHAs + LightEval 0.8.0 |
| `results/measured_serving_confirmation/raw/` | Confirmation GPU-seconds |
| `configs/models/` | **Not** the publication launcher |
| `configs/serving/` | **Not** the publication stack (legacy 0.8.5 YAML) |
| `configs/legacy_models/` | Historical harness JSON |

Manuscript: `paper/main.tex`. ArXiv zip: `paper/arxiv_source.zip`.

Compact campaign JSON omits `finish_reason`, output token IDs, and full traces.
