# Revision summary (2026-09-02)

Convert the frozen 88-run campaign into a publication-ready **stack-pinned measurement study**.

**GPU campaign was not reopened.** No new models, benchmarks, seeds, or result-table edits.

The repository now communicates: *we provide a stack-pinned measurement protocol showing that public quantized reasoning checkpoints can change rank depending on what is measured.* It does not communicate that a quantization method is best.

## Validation performed

| Check | Result |
|-------|--------|
| `python3 scripts/analysis/revision_reanalysis.py --check` | `OK: recomputed report matches .../revision_reanalysis_report.json` |
| `python3 scripts/hpc/qrm_parity/benchmark_serving_confirmation.py --check` | `OK: confirmation raw artifacts present (60 required files)` |
| `python3 scripts/analysis/emit_major_revision_tables.py --check` | `OK: generated tables match .../major_revision_tables.md and .../major_revision_validation.md` |
| `python3 scripts/analysis/measured_serving_confirmation_analysis.py --check` | `OK: recomputed confirmation report matches ... (8 configs)` |
| `python3 scripts/analysis/modal_agreement_analysis.py --check-artifact` | `OK: compact artifact SHA256, 20,000/4,000 structure, T5 accounting, and report internals match.` |
| `python3 scripts/analysis/item_level_descriptive_analysis.py --check` | `OK: item-level descriptive report matches ...` |
| `.venv/bin/python -m pytest tests/ -q` | pass (119 tests) |
| `07_preflight_publication.py --ci` | pass (venv) |
| Frozen MATH-500 / GSM8K / GPQA / serving numbers | unchanged vs prior frozen JSON |

PDF / `arxiv_source.zip` rebuilt 2026-09-02 (`main.pdf` 22 pages; zip SHA256 `420d62c599c03bd7c313c1b2ebb89b2953c7d3c3a98cd2f7f07842b21643d346`).

## Files modified (reason)

### Manuscript

- `paper/main.tex` — three contributions; abstract leads with ranking disagreement; claim audit; FP8 W8A16; Holm-6 / Holm-18 GPQA wording; Condition B cost-accuracy scatter (no Pareto); related work COLM 2025 + RedHatAI note; limitations unevaluated list; CRediT.
- `paper/references.bib` — QRM as COLM 2025; unused `zollo2026` removed.
- `paper/main.md`, `paper/ARTIFACT.md` — aligned with tex / repro commands.

### Protocol and single source of truth

- `scripts/analysis/revision_reanalysis.py` — `load_canonical_pass1()` from frozen report.
- `scripts/analysis/measured_serving_confirmation_analysis.py` — uses that loader (no duplicated `CANONICAL_PASS1`).
- `scripts/analysis/measured_serving_analysis.py` — same.
- `scripts/analysis/emit_major_revision_tables.py` — `--check` compares generated markdown to frozen tables; does not rewrite campaign JSON.
- `scripts/hpc/qrm_parity/benchmark_serving_confirmation.py` — CPU `--check` for raw confirmation artifacts (lazy GPU imports).
- `results/reports/runtime_manifest.json` — dataset SHAs, LightEval 0.8.0, legacy-config warning.
- `results/reports/modal_agreement_report.json` — `lighteval_version: 0.8.0`.

### Config isolation

- `configs/legacy_models/*.json` — historical harness configs (moved).
- `configs/models/README.md` — this directory is **not** the frozen launcher.
- `configs/cells/*.json`, `scripts/pin_hf_revisions.py`, `scripts/hpc/07_preflight_publication.py`, `tests/test_resume_guard.py` — path updates.

### CPU-only descriptive analysis (existing JSON only)

- `scripts/analysis/item_level_descriptive_analysis.py`
- `results/reports/item_level_descriptive_report.json`
- `results/reports/item_level_descriptive_report.md`

### Repro and CI

- `REPRODUCE.md` — expected PASS lines; GPU replay requirements.
- `.github/workflows/ci.yml` — `paper-major-revision`; emit `--check`; modal artifact check; confirmation `--check`; missing-file gate.
- `README.md`, `scripts/analysis/README.md`, `results/README.md`, `CHANGELOG.md`, `AGENTS.md`

## Claim-language confirmations

1. No new GPU experiments.
2. Frozen numeric tables unchanged.
3. FP8 described as Marlin W8A16 on A100, not native W8A8.
4. AWQ findings bound to tested community artifacts.
5. GPQA: Holm-significant within the primary six GPQA contrasts, but not under the joint Holm-18 sensitivity.
6. Cost: aggregate hybrid Cost-of-Pass proxy; GPU-seconds primary; dollars as scenario only.
7. Table 1 (related-work comparison) retained for a JSS-style venue.

## Author note

The CRediT paragraph in `paper/main.tex` is a draft of typical student/advisor roles. Confirm before submission if the split should change.
