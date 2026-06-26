#!/usr/bin/env python3
"""Run inference for one experiment cell (e.g. Level A BF16 MATH-500 seed 0)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset

from src.runners.config_utils import build_prompt, load_cell_config
from src.runners.vllm_runner import build_llm, generate_one


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one inference cell with vLLM.")
    parser.add_argument(
        "--cell-config",
        default="configs/cells/level_a_bf16_seed0.json",
        help="Experiment cell config JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for debugging (e.g. 10).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Default: runs/raw/{cell_id}.jsonl",
    )
    args = parser.parse_args()

    cell = load_cell_config(args.cell_config)
    cell_id = cell["cell_id"]
    out_rel = args.output or f"runs/raw/{cell_id}.jsonl"
    out_path = ROOT / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    task = cell["task"]
    print(f"Loading dataset: {task['dataset_id']} [{task['split']}]")
    dataset = load_dataset(task["dataset_id"], split=task["split"])
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    model_path = cell["model_path"]
    print(f"Loading model from: {model_path}")
    llm = build_llm(model_path, cell["model"])

    rows = []
    total = len(dataset)
    for idx, example in enumerate(dataset):
        problem = example["problem"]
        prompt = build_prompt(task["prompt_template_file"], problem)
        print(f"[{idx + 1}/{total}] generating...")
        result = generate_one(
            llm,
            prompt,
            decoding=cell["decoding"],
            seed=cell["seed"],
            model_path=model_path,
            use_chat_template=cell["model"].get("use_chat_template", True),
        )
        row = {
            "id": example.get("unique_id", str(idx)),
            "problem": problem,
            "gold_solution": example["solution"],
            "subject": example.get("subject"),
            "level": example.get("level"),
            "prompt": result["prompt"],
            "completion": result["completion"],
            "latency_sec": result["latency_sec"],
            "peak_vram_gb": result["peak_vram_gb"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "cell_id": cell_id,
            "model_path": model_path,
            "quant_config": cell["quant_config"],
            "task": task["task_name"],
            "seed": cell["seed"],
        }
        rows.append(row)

        if (idx + 1) % 10 == 0 or idx + 1 == total:
            with out_path.open("w", encoding="utf-8") as f:
                for partial in rows:
                    f.write(json.dumps(partial, ensure_ascii=False) + "\n")
            print(f"checkpoint saved: {out_path} ({len(rows)} rows)")

    print(f"Inference complete. Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
