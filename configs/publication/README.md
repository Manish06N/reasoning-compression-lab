# `configs/publication/` — frozen Paper 1 campaign (pointers only)

This directory is **not a launcher**. It exists so a reviewer who opens `configs/` sees the publication stack before the historical 0.8.5 YAML.

Canonical files (do not duplicate numbers here):

| What | Where |
|------|--------|
| Effective vLLM / decoding / GPU flags | [`../../results/reports/runtime_manifest.json`](../../results/reports/runtime_manifest.json) |
| Dataset IDs and SHAs | [`../tasks/math500.json`](../tasks/math500.json), [`../tasks/gsm8k.json`](../tasks/gsm8k.json), [`../tasks/gpqa_diamond.json`](../tasks/gpqa_diamond.json) |
| Environment lock | [`../../requirements-qrm-paper-vllm070.lock`](../../requirements-qrm-paper-vllm070.lock) |
| Index of those pointers | [`INDEX.json`](INDEX.json) |

Publication stack:

- vLLM **0.7.0** eager
- NVIDIA A100-PCIE-80GB
- QRM `inference.py` after `patches/qrm_hpc_compat.patch`
- FP8 executed as Marlin **W8A16**, not native W8A8

`configs/serving/vllm.yaml` (`version_pin: "0.8.5"`) is historical. `configs/models/` is historical. Do not copy those into a reproduction of Paper 1 numbers.
