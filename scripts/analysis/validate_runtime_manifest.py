#!/usr/bin/env python3
"""Validate the frozen runtime manifest against task configs and campaign counts."""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MANIFEST = os.path.join(REPO, "results", "reports", "runtime_manifest.json")
PUBLICATION_INDEX = os.path.join(REPO, "configs", "publication", "INDEX.json")
TASKS = {
    "MATH-500": os.path.join(REPO, "configs", "tasks", "math500.json"),
    "GSM8K": os.path.join(REPO, "configs", "tasks", "gsm8k.json"),
    "GPQA-Diamond": os.path.join(REPO, "configs", "tasks", "gpqa_diamond.json"),
}


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate and exit nonzero on drift.")
    args = parser.parse_args()
    if not args.check:
        print("Pass --check to validate the frozen runtime manifest.", file=sys.stderr)
        return 2

    if not os.path.isfile(MANIFEST):
        fail(f"missing {MANIFEST}")
    with open(MANIFEST) as fp:
        man = json.load(fp)

    if man.get("n_completions") != 56408:
        fail(f"n_completions={man.get('n_completions')} expected 56408")
    if man.get("n_cells") != 88:
        fail(f"n_cells={man.get('n_cells')} expected 88")
    if man.get("lighteval_version") != "0.8.0":
        fail("lighteval_version must be 0.8.0")

    shared = man.get("shared") or {}
    if shared.get("vllm_version") != "0.7.0":
        fail("shared.vllm_version must be 0.7.0")
    if shared.get("enforce_eager") is not True:
        fail("shared.enforce_eager must be true")
    if shared.get("gpu_model") != "NVIDIA A100-PCIE-80GB":
        fail("shared.gpu_model mismatch")
    if shared.get("torch") != "2.5.1":
        fail("shared.torch must be 2.5.1")
    if shared.get("cuda_toolkit") != "12.4":
        fail("shared.cuda_toolkit must be 12.4")
    if "nvidia_driver" not in shared:
        fail("shared.nvidia_driver must be present (UNRECORDED is allowed)")
    if shared.get("nvidia_driver") == "UNRECORDED":
        note = shared.get("nvidia_driver_note") or ""
        if "unavailable" not in note.lower():
            fail(
                "UNRECORDED nvidia_driver requires nvidia_driver_note "
                "explaining that the driver version was unavailable"
            )
    required_cell_keys = {"hf_id", "revision", "weight_format"}
    for c in man.get("cells") or []:
        missing = required_cell_keys - set(c)
        if missing:
            fail(f"cell missing {sorted(missing)}: {c.get('hf_id')}")

    datasets = man.get("datasets") or {}
    for name, task_path in TASKS.items():
        with open(task_path) as fp:
            task = json.load(fp)
        rec = datasets.get(name) or {}
        if rec.get("revision") != task.get("revision"):
            fail(f"{name} SHA {rec.get('revision')} != {task.get('revision')}")
        if rec.get("dataset_id") != task.get("dataset_id"):
            fail(f"{name} dataset_id mismatch")

    cells = man.get("cells") or []
    if len(cells) != 8:
        fail(f"expected 8 checkpoint cells, got {len(cells)}")
    fp8 = [c for c in cells if c.get("weight_format") == "FP8"]
    if len(fp8) != 2:
        fail("expected two FP8 cells")
    for c in fp8:
        note = c.get("a100_execution", "")
        if "W8A16" not in note or "W8A8" not in note:
            fail(f"FP8 cell missing W8A16/not-W8A8 note: {c.get('hf_id')}")

    if not os.path.isfile(PUBLICATION_INDEX):
        fail(f"missing {PUBLICATION_INDEX}")
    with open(PUBLICATION_INDEX) as fp:
        idx = json.load(fp)
    if idx.get("vllm_version") != shared.get("vllm_version"):
        fail("publication INDEX vllm_version does not match runtime_manifest")
    if idx.get("enforce_eager") != shared.get("enforce_eager"):
        fail("publication INDEX enforce_eager does not match runtime_manifest")
    if idx.get("gpu_model") != shared.get("gpu_model"):
        fail("publication INDEX gpu_model does not match runtime_manifest")
    if idx.get("lighteval_version") != man.get("lighteval_version"):
        fail("publication INDEX lighteval_version does not match runtime_manifest")
    for rel in idx.get("tasks") or []:
        path = os.path.join(REPO, rel)
        if not os.path.isfile(path):
            fail(f"publication INDEX missing task file {rel}")

    print(
        "OK: runtime_manifest.json matches campaign counts, "
        "LightEval 0.8.0, vLLM 0.7.0, task SHAs, and configs/publication/INDEX.json."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
