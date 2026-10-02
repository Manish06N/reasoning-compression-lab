# Licensing and third-party notice

## This repository

| Part | License |
|---|---|
| Source code (`src/`, `scripts/`, `tests/`, `slurm/`, `configs/`, `patches/` as authored here) | MIT, see [`LICENSE`](LICENSE) |
| Manuscript text and figures (`paper/`), derived result files (`results/`), documentation (`docs/`) | Creative Commons Attribution 4.0 International (CC BY 4.0), <https://creativecommons.org/licenses/by/4.0/> |

Please cite the paper (see [`CITATION.cff`](CITATION.cff)) when you reuse the results.

## What is not redistributed here

* **Model weights.** The eight evaluated checkpoints are public Hugging Face uploads (IDs and revisions in the paper, Appendix "Checkpoints"). Their own licenses apply (DeepSeek-R1-Distill models, the RedHatAI quantized artifacts, and the community `jakiAJK` AWQ uploads). Check each model card before reuse.
* **Benchmark item text.** GPQA-Diamond is a gated dataset and its question text is deliberately never stored in this repository (compact result files hold only per-row scores, token counts and flags). MATH-500 and GSM8K are public; fetch them from their original sources.
* **Full chain-of-thought traces** are not released (size and benchmark-access terms). The compact per-completion records and the recovered MATH-500 answer strings in `results/` are sufficient for every analysis in the paper except re-scoring.

## Third-party code this work builds on (not vendored)

* **Quantized-Reasoning-Models (QRM)**, MIT License, Copyright (c) 2025 ruikangliu: the official evaluation harness (`inference.py`, `lighteval_custom/`) used for the campaign. It is cloned into `external/` by `scripts/hpc/qrm_parity/setup_official_qrm_repo.sh` and is not part of this repository.
* **LightEval**, **vLLM** and **math-verify** are used as installed packages (not vendored). The two small compatibility patches in `patches/` modify QRM and LightEval sources and therefore carry those projects' MIT licenses.

If you find a licensing problem, please open an issue.
