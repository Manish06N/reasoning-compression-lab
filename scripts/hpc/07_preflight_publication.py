#!/usr/bin/env python3
"""CPU-side preflight for the HPC publication blocks.

This intentionally avoids constructing a vLLM engine. The GPU smoke test covers
that path. This script catches the common cheap failures first: missing configs,
wrong machine split, missing model folders, prompt formatting, and dataset access.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset

from src.runners.config_utils import build_prompt, load_cell_config


EXPECTED_HPC_BLOCKS = {
    "b01_parallel_bf16_anchors.sh",
    "b02_parallel_fp8.sh",
    "b03_parallel_awq4.sh",
    "b04_parallel_gptq4.sh",
    "b05_single_gptq3.sh",
    "b06_single_gsm8k.sh",
    "b08_qwen15b_bf16_fp8.sh",
    "b09_qwen15b_awq4_gptq4.sh",
}


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def block_cells(block_file: Path) -> list[str]:
    cells: list[str] = []
    for line in block_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip().strip('"')
        if ":" in stripped and stripped.endswith(".json"):
            cells.append(stripped.split(":", 1)[1])
    return cells


def check_static() -> None:
    print("== static checks ==")
    run(
        [
            "bash",
            "-n",
            "scripts/hpc/submit_hpc_blocks.sh",
            "scripts/hpc/run_hpc_2a100_publication.sh",
            "slurm/hpc_2a100_b01_parallel.slurm",
            "slurm/hpc_2a100_b07_gpqa.slurm",
        ]
    )
    run(["python", "-m", "compileall", "-q", "scripts", "src"])


def check_prompt() -> None:
    print("== prompt formatting ==")
    prompt = build_prompt("prompts/math500.txt", question="2+2?")
    if "{ANSWER}" not in prompt:
        fail("math prompt did not preserve literal {ANSWER}")
    if "2+2?" not in prompt:
        fail("math prompt did not include question")
    print("prompt formatting ok")


def check_blocks_and_models() -> None:
    print("== block/config/model wiring ==")
    block_dir = ROOT / "configs/machine_split/hpc_blocks"
    present = {p.name for p in block_dir.glob("*.sh")}
    missing_blocks = sorted(EXPECTED_HPC_BLOCKS - present)
    if missing_blocks:
        fail(f"missing expected HPC blocks: {missing_blocks}")

    checked_cells = 0
    for block_name in sorted(EXPECTED_HPC_BLOCKS):
        block_file = block_dir / block_name
        cells = block_cells(block_file)
        if not cells:
            fail(f"{block_name} has no cells")
        print(block_name)
        for cell_rel in cells:
            cell = load_cell_config(cell_rel)
            model_path = Path(cell["model_path"])
            task_name = cell["task"]["task_name"]
            print(f"  {cell['cell_id']} -> {model_path} task={task_name}")
            if not model_path.exists():
                fail(f"missing model path for {cell['cell_id']}: {model_path}")
            if not (model_path / "config.json").exists():
                fail(f"missing config.json in {model_path}")
            if not any((model_path / name).exists() for name in ("tokenizer.json", "tokenizer.model")):
                fail(f"missing tokenizer in {model_path}")
            weight_count = sum(
                1
                for p in model_path.rglob("*")
                if p.is_file() and p.name.endswith((".safetensors", ".bin", ".pt"))
            )
            if weight_count == 0:
                fail(f"missing model weights in {model_path}")
            checked_cells += 1
    print(f"checked {checked_cells} HPC cell entries")


def check_datasets() -> None:
    print("== dataset access ==")
    for task_rel in ("configs/tasks/math500.json", "configs/tasks/gsm8k.json"):
        task = json.loads((ROOT / task_rel).read_text(encoding="utf-8"))
        if task.get("config_name"):
            dataset = load_dataset(task["dataset_id"], task["config_name"], split=task["split"])
        else:
            dataset = load_dataset(task["dataset_id"], split=task["split"])
        print(
            task["task_name"],
            task["dataset_id"],
            task.get("config_name"),
            task["split"],
            len(dataset),
            dataset.column_names,
        )
        if task["task_name"] == "math500" and len(dataset) != 500:
            fail("MATH-500 row count is not 500")
        if task["task_name"] == "gsm8k" and len(dataset) != 1319:
            fail("GSM8K test row count is not 1319")


def main() -> None:
    check_static()
    check_prompt()
    check_blocks_and_models()
    check_datasets()
    print("HPC publication CPU preflight passed.")


if __name__ == "__main__":
    main()
