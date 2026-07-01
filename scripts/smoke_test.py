#!/usr/bin/env python3
"""Tiny vLLM smoke test on 1-3 math questions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runners.config_utils import build_prompt, load_cell_config
from src.runners.vllm_runner import build_llm, generate_one


SMOKE_QUESTIONS = [
    "What is 17 + 28?",
    "Solve for x: 2x + 5 = 19.",
    "A rectangle has width 4 and height 9. What is its area?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny BF16 vLLM smoke test.")
    parser.add_argument(
        "--cell-config",
        default="configs/cells/level_a_bf16_seed0.json",
        help="Experiment cell config JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of smoke questions to run.",
    )
    parser.add_argument(
        "--output",
        default="runs/raw/smoke_test.jsonl",
        help="Output JSONL path relative to repo root.",
    )
    parser.add_argument(
        "--decoding-config",
        default="configs/decoding/smoke_debug.yaml",
        help="Decoding YAML for smoke test (default: 1024 max tokens).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional override for max_tokens.",
    )
    args = parser.parse_args()

    cell = load_cell_config(args.cell_config)
    if args.decoding_config:
        from src.runners.config_utils import load_decoding_from_file

        cell["decoding"] = load_decoding_from_file(args.decoding_config)
    if args.max_tokens is not None:
        cell["decoding"]["max_tokens"] = args.max_tokens

    print(f"Smoke decoding: max_tokens={cell['decoding']['max_tokens']}")
    model_path = cell["model_path"]
    print(f"Loading model from: {model_path}")
    llm = build_llm(model_path, cell["model"])

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, question in enumerate(SMOKE_QUESTIONS[: args.limit]):
        prompt = build_prompt(cell["task"]["prompt_template_file"], question=question)
        print(f"\n[{idx + 1}/{args.limit}] Generating...")
        result = generate_one(
            llm,
            prompt,
            decoding=cell["decoding"],
            seed=cell["seed"],
            model_path=model_path,
            use_chat_template=cell["model"].get("use_chat_template", True),
        )
        row = {
            "id": f"smoke_{idx}",
            "problem": question,
            "completion": result["completion"],
            "latency_sec": result["latency_sec"],
            "peak_vram_gb": result["peak_vram_gb"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "model_path": model_path,
            "quant_config": cell["quant_config"],
            "seed": cell["seed"],
        }
        rows.append(row)
        print(f"latency={row['latency_sec']:.1f}s vram={row['peak_vram_gb']:.2f}GB")
        print(f"completion preview: {row['completion'][:300]}...")

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSmoke test complete. Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
