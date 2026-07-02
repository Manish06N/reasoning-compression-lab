#!/usr/bin/env python3
"""Run maj@k inference: N vLLM samples per problem (Level B calibration pilot)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runners.checkpoint_utils import atomic_write_jsonl, backup_file, write_progress
from src.runners.config_utils import build_prompt, load_cell_config, load_decoding_from_file
from src.runners.dataset_rows import prepare_example_row
from src.runners.inference_session import (
    guard_and_recover_resume,
    load_task_dataset,
    setup_output_paths,
)
from src.runners.publication_mode import assert_clean_git_tree, is_publication_mode
from src.runners.raw_row import build_raw_response_row
from src.runners.resume_guard import allow_resume_from_env
from src.runners.run_spec import run_spec_from_cell
from src.runners.sampling_utils import sample_seed_for_draw
from src.runners.vllm_runner import build_llm, generate_one
from src.schemas.provenance import provenance_fields
from src.schemas.validate import SchemaValidationError, validate_jsonl_rows

CHECKPOINT_EVERY = 10


def _validate_checkpoint(out_path: Path, *, publication: bool) -> None:
    if publication:
        validation = validate_jsonl_rows(out_path, every_nth=1)
    else:
        validation = validate_jsonl_rows(out_path, limit=3)
    if not validation["valid"]:
        msg = f"schema validation: {validation['errors'][:3]}"
        if publication:
            raise SchemaValidationError(msg)
        print(f"WARN: {msg}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-sample inference for maj@k / calibration.")
    parser.add_argument("--cell-config", default="configs/cells/level_a_bf16_seed0.json")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--decoding-config", default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--allow-resume", action="store_true")
    parser.add_argument("--publication", action="store_true")
    args = parser.parse_args()

    publication = is_publication_mode(cli_flag=args.publication)
    if publication:
        assert_clean_git_tree(ROOT)

    cell = load_cell_config(args.cell_config)
    cell_id = cell["cell_id"]
    n_samples = max(1, args.n_samples)
    base_seed = int(cell.get("seed", 0))

    out_path, archive_root, backup_root = setup_output_paths(
        cell_id,
        args.output,
        fresh=args.fresh,
        suffix=f"_maj{n_samples}",
    )

    if args.decoding_config:
        cell["decoding"] = load_decoding_from_file(args.decoding_config)
    if args.max_model_len is not None:
        cell["model"] = dict(cell["model"])
        cell["model"]["max_model_len"] = args.max_model_len

    task = cell["task"]
    prompt_template_file = task["prompt_template_file"]
    run_spec = run_spec_from_cell(
        cell,
        prompt_template_file=prompt_template_file,
        batch_size=1,
        n_samples=n_samples,
        max_model_len=cell["model"].get("max_model_len"),
        publication_mode=publication,
    )

    dataset = load_task_dataset(cell, args.limit)
    allow_resume = args.allow_resume or allow_resume_from_env()
    rows = guard_and_recover_resume(
        out_path,
        cell,
        allow_resume=allow_resume,
        backup_root=backup_root,
        run_spec=run_spec,
    )

    completed_pairs = {(r["id"], r.get("sample_index", 0)) for r in rows}
    total = len(dataset)
    expected_rows = total * n_samples
    print(f"Cell {cell_id}: {n_samples} samples × {total} problems = {expected_rows} rows")

    model_path = cell["model_path"]
    use_chat = cell["model"].get("use_chat_template", True)
    llm = build_llm(model_path, cell["model"])
    checkpoint_every = max(1, args.checkpoint_every)
    run_provenance = provenance_fields(cell, run_spec=run_spec)

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
            row = build_raw_response_row(
                row_base=row_base,
                result=result,
                cell=cell,
                prompt_template_file=prompt_template_file,
                run_provenance=run_provenance,
                batch_size=1,
                sample_index=sample_index,
                sample_seed=draw_seed,
                n_samples=n_samples,
                telemetry_method="measured",
                decoding_config_override=args.decoding_config,
            )
            rows.append(row)
            completed_pairs.add((item_id, sample_index))

            if len(rows) % checkpoint_every == 0:
                atomic_write_jsonl(out_path, rows)
                _validate_checkpoint(out_path, publication=publication)
                if backup_root:
                    backup_file(out_path, backup_root, "raw")
                if archive_root:
                    write_progress(
                        archive_root, cell_id, len(rows), expected_rows, status="in_progress"
                    )

    atomic_write_jsonl(out_path, rows)
    _validate_checkpoint(out_path, publication=publication)
    if archive_root:
        write_progress(archive_root, cell_id, len(rows), expected_rows, status="completed")
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
