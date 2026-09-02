# `configs/serving/` is not the frozen publication stack

**DO NOT USE FOR PAPER REPRODUCTION.**

These YAML files (`vllm.yaml`, `sglang.yaml`, `llamacpp.yaml`) belong to the **historical local harness**. `vllm.yaml` still says `version_pin: "0.8.5"`. That is **not** the 88-run campaign.

Publication serving:

- Engine: QRM `inference.py` after `patches/qrm_hpc_compat.patch`
- vLLM **0.7.0** eager on NVIDIA A100-PCIE-80GB
- Recorded in [`../../results/reports/runtime_manifest.json`](../../results/reports/runtime_manifest.json)
- Confirmation protocol: [`../../docs/MEASURED_SERVING_CONFIRMATION_PROTOCOL.md`](../../docs/MEASURED_SERVING_CONFIRMATION_PROTOCOL.md)

Do not copy these YAML files into a reviewer reproduction of Paper 1 numbers.
