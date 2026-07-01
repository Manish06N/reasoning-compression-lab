#!/usr/bin/env python3
"""Run inference for one experiment cell (e.g. Level A BF16 MATH-500 seed 0)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset

from src.runners.checkpoint_utils import (
    atomic_write_jsonl,
    backup_file,
    load_jsonl,
    recover_jsonl_from_backup,
    update_state,
    validate_jsonl,
    write_progress,
)
from src.runners.config_utils import build_prompt, load_cell_config, load_decoding_from_file
from src.runners.vllm_runner import build_llm, generate_chunk

CHECKPOINT_EVERY = 10


def _output_root_for(out_path: Path) -> Path | None:
    """Infer archive root (parent of raw/) when output lives under .../raw/."""
    if out_path.parent.name == "raw":
        return out_path.parent.parent
    return None


def _get_field(example, *names: str) -> str:
    for name in names:
        if name in example and example[name] is not None:
            return str(example[name])
    return ""


def _prepare_example_row(example, task: dict, cell: dict, global_i: int) -> tuple[dict, dict]:
    task_name = task["task_name"]
    if task_name.startswith("gpqa"):
        question = _get_field(example, "Question", "question", "problem")
        correct = _get_field(example, "Correct Answer", "correct_answer", "answer")
        distractors = [
            _get_field(example, "Incorrect Answer 1", "incorrect_answer_1"),
            _get_field(example, "Incorrect Answer 2", "incorrect_answer_2"),
            _get_field(example, "Incorrect Answer 3", "incorrect_answer_3"),
        ]
        choices = [(correct, "correct")] + [(d, "incorrect") for d in distractors]
        rng = random.Random(int(cell.get("seed", 0)) * 1_000_003 + global_i)
        rng.shuffle(choices)
        letters = "ABCD"
        prompt_fields = {"question": question}
        gold_letter = None
        for letter, (choice, kind) in zip(letters, choices):
            prompt_fields[letter.lower()] = choice
            if kind == "correct":
                gold_letter = letter
        row_base = {
            "id": example.get("Record ID", example.get("unique_id", str(global_i))),
            "problem": question,
            "choices": {letter: prompt_fields[letter.lower()] for letter in letters},
            "gold_letter": gold_letter,
            "gold_answer": correct,
        }
        return prompt_fields, row_base

    problem_field = task.get("problem_field", "problem")
    solution_field = task.get("solution_field", "solution")
    problem = str(example[problem_field])
    return {"question": problem.strip()}, {
        "id": example.get("unique_id", str(global_i)),
        "problem": problem,
        "gold_solution": example[solution_field],
        "subject": example.get("subject"),
        "level": example.get("level"),
    }


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
    parser.add_argument(
        "--decoding-config",
        default=None,
        help="Override cell decoding (YAML), e.g. configs/decoding/pilot_5080.yaml",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override model max_model_len for this run (pilot: 8192).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Prompts per vLLM.generate() call (5080 pilot: 4 for 1.5B, 2 for 7B/8B).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_EVERY,
        help="Atomic checkpoint + backup every N completed rows.",
    )
    args = parser.parse_args()

    cell = load_cell_config(args.cell_config)
    cell_id = cell["cell_id"]
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
    else:
        out_path = ROOT / f"runs/raw/{cell_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    archive_root = _output_root_for(out_path)
    backup_root = (archive_root / "_backup") if archive_root else None

    if args.decoding_config:
        cell["decoding"] = load_decoding_from_file(args.decoding_config)
    if args.max_model_len is not None:
        cell["model"] = dict(cell["model"])
        cell["model"]["max_model_len"] = args.max_model_len
    elif cell["decoding"].get("max_model_len"):
        cell["model"] = dict(cell["model"])
        cell["model"]["max_model_len"] = int(cell["decoding"]["max_model_len"])

    task = cell["task"]
    print(f"Loading dataset: {task['dataset_id']} [{task['split']}]")
    if task.get("config_name"):
        dataset = load_dataset(task["dataset_id"], task["config_name"], split=task["split"])
    else:
        dataset = load_dataset(task["dataset_id"], split=task["split"])
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    ok, _ = validate_jsonl(out_path)
    if not ok:
        print(f"WARN: corrupt JSONL {out_path} — attempting restore from backup")
        if backup_root and recover_jsonl_from_backup(out_path, backup_root):
            print(f"Restored {out_path} from _backup/latest/raw/")
        else:
            corrupt = out_path.with_suffix(out_path.suffix + ".corrupt")
            out_path.replace(corrupt)
            print(f"Moved corrupt file → {corrupt}; starting fresh")

    rows = load_jsonl(out_path)
    start_idx = len(rows)
    total = len(dataset)
    if start_idx:
        print(f"Resuming {cell_id}: {start_idx}/{total} rows already in {out_path}")
    if start_idx >= total:
        print(f"Already complete ({start_idx}/{total} rows).")
        if archive_root:
            write_progress(archive_root, cell_id, start_idx, total, status="completed")
        return

    model_path = cell["model_path"]
    batch_size = max(1, args.batch_size)
    print(f"Loading model from: {model_path}")
    print(
        f"Decoding: max_tokens={cell['decoding'].get('max_tokens')}, "
        f"max_model_len={cell['model'].get('max_model_len')}, batch_size={batch_size}"
    )
    if archive_root:
        update_state(
            archive_root,
            last_cell_id=cell_id,
            last_phase="inference",
            rows_done=start_idx,
            rows_total=total,
        )
        write_progress(archive_root, cell_id, start_idx, total, status="in_progress")

    llm = build_llm(model_path, cell["model"])
    use_chat = cell["model"].get("use_chat_template", True)
    checkpoint_every = max(1, args.checkpoint_every)

    idx = start_idx
    while idx < total:
        batch_end = min(idx + batch_size, total)
        batch_examples = [dataset[i] for i in range(idx, batch_end)]
        prepared = [
            _prepare_example_row(example, task, cell, global_i)
            for global_i, example in enumerate(batch_examples, start=idx)
        ]
        prompts = [
            build_prompt(task["prompt_template_file"], **prompt_fields)
            for prompt_fields, _ in prepared
        ]
        print(f"[{idx + 1}-{batch_end}/{total}] generating batch of {len(prompts)}...")
        results = generate_chunk(
            llm,
            prompts,
            decoding=cell["decoding"],
            seed=cell["seed"],
            model_path=model_path,
            use_chat_template=use_chat,
        )
        for (_, row_base), result in zip(prepared, results):
            row = {
                **row_base,
                "prompt": result["prompt"],
                "completion": result["completion"],
                "latency_sec": result["latency_sec"],
                "time_to_first_token_sec": result.get("time_to_first_token_sec"),
                "peak_vram_gb": result["peak_vram_gb"],
                "vram_before_gb": result.get("vram_before_gb"),
                "vram_after_gb": result.get("vram_after_gb"),
                "vram_max_gb": result.get("vram_max_gb"),
                "gpu_util_mean": result.get("gpu_util_mean"),
                "gpu_util_max": result.get("gpu_util_max"),
                "power_watts_mean": result.get("power_watts_mean"),
                "power_watts_max": result.get("power_watts_max"),
                "energy_joules": result.get("energy_joules"),
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result.get("total_tokens"),
                "tokens_per_second": result.get("tokens_per_second"),
                "decode_tokens_per_second": result.get("decode_tokens_per_second"),
                "seconds_per_output_token": result.get("seconds_per_output_token"),
                "tokens_per_joule": result.get("tokens_per_joule"),
                "finish_reason": result.get("finish_reason"),
                "stop_reason": result.get("stop_reason"),
                "truncated": result.get("truncated"),
                "completion_chars": result.get("completion_chars"),
                "cell_id": cell_id,
                "model_path": model_path,
                "quant_config": cell["quant_config"],
                "task": task["task_name"],
                "seed": cell["seed"],
                "batch_size": batch_size,
                "decoding_temperature": cell["decoding"].get("temperature"),
                "decoding_top_p": cell["decoding"].get("top_p"),
                "decoding_max_tokens": cell["decoding"].get("max_tokens"),
                "decoding_repetition_penalty": cell["decoding"].get("repetition_penalty"),
                "max_model_len": cell["model"].get("max_model_len"),
            }
            if args.decoding_config:
                row["decoding_config"] = args.decoding_config
            rows.append(row)

        idx = batch_end
        if len(rows) % checkpoint_every == 0 or idx == total:
            atomic_write_jsonl(out_path, rows)
            print(f"checkpoint saved: {out_path} ({len(rows)} rows)")
            if backup_root:
                backup_file(out_path, backup_root, "raw")
                if archive_root:
                    write_progress(archive_root, cell_id, len(rows), total, status="in_progress")
                    update_state(
                        archive_root,
                        last_cell_id=cell_id,
                        last_phase="inference",
                        rows_done=len(rows),
                        rows_total=total,
                    )

    if archive_root:
        write_progress(archive_root, cell_id, len(rows), total, status="completed")
        update_state(
            archive_root,
            last_cell_id=cell_id,
            last_phase="inference_complete",
            rows_done=len(rows),
            rows_total=total,
        )

    print(f"Inference complete. Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
