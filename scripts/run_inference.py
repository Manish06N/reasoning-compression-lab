#!/usr/bin/env python3
"""Run inference for one experiment cell (e.g. Level A BF16 MATH-500 seed 0)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runners.checkpoint_utils import (
    atomic_write_jsonl,
    backup_file,
    update_state,
    write_progress,
)
from src.runners.config_utils import build_prompt, load_cell_config, load_decoding_from_file
from src.runners.dataset_rows import prepare_example_row
from src.runners.inference_session import (
    ConfigurationError,
    assert_publication_batch_size,
    guard_and_recover_resume,
    load_task_dataset,
    setup_output_paths,
)
from src.runners.publication_mode import assert_clean_git_tree, is_publication_mode
from src.runners.raw_row import build_raw_response_row
from src.runners.resume_guard import allow_resume_from_env
from src.runners.run_spec import run_spec_from_cell
from src.runners.vllm_runner import build_llm, generate_chunk
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
    parser = argparse.ArgumentParser(description="Run one inference cell with vLLM.")
    parser.add_argument("--cell-config", default="configs/cells/level_a_bf16_seed0.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--decoding-config", default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--allow-resume", action="store_true")
    parser.add_argument(
        "--publication",
        action="store_true",
        help="Publication mode: require batch_size=1 (also honors QREASON_PUBLICATION_MODE).",
    )
    args = parser.parse_args()

    publication = is_publication_mode(cli_flag=args.publication)
    if publication:
        assert_clean_git_tree(ROOT)

    cell = load_cell_config(args.cell_config)
    cell_id = cell["cell_id"]
    batch_size = max(1, args.batch_size)

    try:
        assert_publication_batch_size(batch_size, publication=args.publication or publication)
    except ConfigurationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    out_path, archive_root, backup_root = setup_output_paths(
        cell_id, args.output, fresh=args.fresh
    )

    if args.decoding_config:
        cell["decoding"] = load_decoding_from_file(args.decoding_config)
    if args.max_model_len is not None:
        cell["model"] = dict(cell["model"])
        cell["model"]["max_model_len"] = args.max_model_len
    elif cell["decoding"].get("max_model_len"):
        cell["model"] = dict(cell["model"])
        cell["model"]["max_model_len"] = int(cell["decoding"]["max_model_len"])

    task = cell["task"]
    prompt_template_file = task["prompt_template_file"]
    run_spec = run_spec_from_cell(
        cell,
        prompt_template_file=prompt_template_file,
        batch_size=batch_size,
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
    telemetry_method = "equal_split" if batch_size > 1 else "measured"
    print(f"Loading model from: {model_path}")
    print(
        f"Decoding: temperature={cell['decoding'].get('temperature')}, "
        f"top_p={cell['decoding'].get('top_p')}, "
        f"max_tokens={cell['decoding'].get('max_tokens')}, "
        f"repetition_penalty={cell['decoding'].get('repetition_penalty')}, "
        f"seed={cell['seed']}, "
        f"max_model_len={cell['model'].get('max_model_len')}, batch_size={batch_size}, "
        f"telemetry_method={telemetry_method}"
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
    run_provenance = provenance_fields(cell, run_spec=run_spec)

    idx = start_idx
    while idx < total:
        batch_end = min(idx + batch_size, total)
        batch_examples = [dataset[i] for i in range(idx, batch_end)]
        prepared = [
            prepare_example_row(example, task, cell, global_i)
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
            row = build_raw_response_row(
                row_base=row_base,
                result=result,
                cell=cell,
                prompt_template_file=prompt_template_file,
                run_provenance=run_provenance,
                batch_size=batch_size,
                telemetry_method=telemetry_method,
                decoding_config_override=args.decoding_config,
            )
            rows.append(row)

        idx = batch_end
        if len(rows) % checkpoint_every == 0 or idx == total:
            atomic_write_jsonl(out_path, rows)
            _validate_checkpoint(out_path, publication=publication)
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
