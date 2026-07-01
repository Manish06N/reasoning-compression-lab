#!/usr/bin/env python3
"""Run maj@k inference: N vLLM samples per problem (Level B calibration pilot).

Adapted from Calibrating-LLMs-with-Consistency predict.py flow + sober-reasoning seed pattern.
Each sample uses seed = base_seed * 1000 + sample_index (see src/runners/sampling_utils.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse dataset prep from src.runners.dataset_rows
from src.runners.dataset_rows import output_root_for, prepare_example_row
from src.runners.checkpoint_utils import atomic_write_jsonl, backup_file, load_jsonl, write_progress
from src.runners.config_utils import build_prompt, load_cell_config, load_decoding_from_file
from src.runners.sampling_utils import sample_seed_for_draw
from src.runners.vllm_runner import build_llm, generate_one

CHECKPOINT_EVERY = 10


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-sample inference for maj@k / calibration.")
    parser.add_argument("--cell-config", default="configs/cells/level_a_bf16_seed0.json")
    parser.add_argument("--n-samples", type=int, default=5, help="Samples per problem (maj@k).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--decoding-config", default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    args = parser.parse_args()

    from datasets import load_dataset

    cell = load_cell_config(args.cell_config)
    cell_id = cell["cell_id"]
    n_samples = max(1, args.n_samples)
    base_seed = int(cell.get("seed", 0))

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
    else:
        out_path = ROOT / f"runs/raw/{cell_id}_maj{n_samples}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.decoding_config:
        cell["decoding"] = load_decoding_from_file(args.decoding_config)
    if args.max_model_len is not None:
        cell["model"] = dict(cell["model"])
        cell["model"]["max_model_len"] = args.max_model_len

    task = cell["task"]
    if task.get("config_name"):
        dataset = load_dataset(task["dataset_id"], task["config_name"], split=task["split"])
    else:
        dataset = load_dataset(task["dataset_id"], split=task["split"])
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    rows = load_jsonl(out_path)
    completed_pairs = {(r["id"], r.get("sample_index", 0)) for r in rows}
    total = len(dataset)
    expected_rows = total * n_samples
    print(f"Cell {cell_id}: {n_samples} samples × {total} problems = {expected_rows} rows")
    print(
        f"Decoding: repetition_penalty={cell['decoding'].get('repetition_penalty')}, "
        f"base_seed={base_seed}"
    )

    archive_root = output_root_for(out_path)
    backup_root = (archive_root / "_backup") if archive_root else None

    model_path = cell["model_path"]
    use_chat = cell["model"].get("use_chat_template", True)
    llm = build_llm(model_path, cell["model"])
    checkpoint_every = max(1, args.checkpoint_every)

    for global_i in range(total):
        example = dataset[global_i]
        prompt_fields, row_base = prepare_example_row(example, task, cell, global_i)
        prompt = build_prompt(task["prompt_template_file"], **prompt_fields)
        item_id = row_base["id"]

        for sample_index in range(n_samples):
            if (item_id, sample_index) in completed_pairs:
                continue
            draw_seed = sample_seed_for_draw(base_seed, sample_index)
            print(
                f"[{global_i + 1}/{total}] sample {sample_index + 1}/{n_samples} "
                f"id={item_id} seed={draw_seed}"
            )
            result = generate_one(
                llm,
                prompt,
                decoding=cell["decoding"],
                seed=draw_seed,
                model_path=model_path,
                use_chat_template=use_chat,
            )
            row = {
                **row_base,
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
                "seed": base_seed,
                "sample_index": sample_index,
                "sample_seed": draw_seed,
                "n_samples": n_samples,
                "decoding_repetition_penalty": cell["decoding"].get("repetition_penalty"),
            }
            for field in (
                "finish_reason",
                "truncated",
                "time_to_first_token_sec",
                "energy_joules",
                "vram_max_gb",
            ):
                if field in result:
                    row[field] = result[field]
            rows.append(row)
            completed_pairs.add((item_id, sample_index))

            if len(rows) % checkpoint_every == 0:
                atomic_write_jsonl(out_path, rows)
                if backup_root:
                    backup_file(out_path, backup_root, "raw")
                if archive_root:
                    write_progress(archive_root, cell_id, len(rows), expected_rows, status="in_progress")

    atomic_write_jsonl(out_path, rows)
    if archive_root:
        write_progress(archive_root, cell_id, len(rows), expected_rows, status="completed")
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
