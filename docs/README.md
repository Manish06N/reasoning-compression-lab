# Documentation index

Start with the top-level [`README.md`](../README.md), [`REPRODUCE.md`](../REPRODUCE.md) and the paper-to-code map [`PAPER_TO_CODE.md`](PAPER_TO_CODE.md).

## Method and protocol notes

| Doc | Purpose |
|---|---|
| [ANSWER_NORMALIZATION.md](ANSWER_NORMALIZATION.md) | Answer extraction and equivalence used for scoring (LightEval 0.8.0 + math-verify) |
| [MEASURED_SERVING_CONFIRMATION_PROTOCOL.md](MEASURED_SERVING_CONFIRMATION_PROTOCOL.md) | Controlled serving confirmation (Conditions A and B, repeats, CV rule) |
| [MEASURED_SERVING_PROTOCOL.md](MEASURED_SERVING_PROTOCOL.md) | Earlier unconstrained serving timing (provenance only) |
| [MODEL_ROSTER.md](MODEL_ROSTER.md), [MODEL_SCOPE_DECISION.md](MODEL_SCOPE_DECISION.md) | Which checkpoints are evaluated and why |
| [GPQA_ACCESS.md](GPQA_ACCESS.md) | GPQA-Diamond access terms; item text is never stored in this repository |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | Known limitations and pitfalls |
| [EXPERIMENTAL_PARAMETERS_AND_OUTPUT_AUDIT.md](EXPERIMENTAL_PARAMETERS_AND_OUTPUT_AUDIT.md) | Decoding parameters and output audit |
| [QRM_STACK_PARITY_AUDIT.md](QRM_STACK_PARITY_AUDIT.md), [QRM_OFFICIAL_HPC_TROUBLESHOOTING.md](QRM_OFFICIAL_HPC_TROUBLESHOOTING.md) | Parity with the official QRM harness and environment setup |
| [HARDWARE_POLICY.md](HARDWARE_POLICY.md), [HPC_PARAM_RUDRA.md](HPC_PARAM_RUDRA.md), [PARAM_RUDRA_SLURM.md](PARAM_RUDRA_SLURM.md), [ENV_VARS.md](ENV_VARS.md), [GPTQ4_PREP.md](GPTQ4_PREP.md), [RUNBOOK.md](RUNBOOK.md) | Cluster and run-time details for the GPU campaign |
| [PAPER1_DESIGN.md](PAPER1_DESIGN.md), [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md), [REPO_MAP.md](REPO_MAP.md) | Design notes and the code layout (older notes; the top-level README is current) |

## Archive

[`archive/`](archive/) holds historical material kept for provenance: project process notes and audit write-ups (`process_notes/`), planning and venue notes (`planning/`), cluster set-up guides (`hpc_misc/`), the original README, and the pre-2026-09-22 changelog. Nothing in the paper depends on it.
