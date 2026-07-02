"""Shared inference session helpers for run_inference and multisample scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from src.runners.checkpoint_utils import load_jsonl, recover_jsonl_from_backup, validate_jsonl
from src.runners.config_utils import REPO_ROOT
from src.runners.dataset_rows import output_root_for
from src.runners.resume_guard import resume_block_reason
from src.runners.revision_resolver import load_dataset_with_revision


class ConfigurationError(Exception):
    """Invalid run configuration (publication mode, batch size, etc.)."""


def assert_publication_batch_size(batch_size: int, *, publication: bool) -> None:
    """Fail closed when publication mode requires batch_size=1."""
    pub_env = os.environ.get("QREASON_PUBLICATION_MODE", "").strip().lower() in ("1", "true", "yes")
    batch_invariant = os.environ.get("VLLM_BATCH_INVARIANT", "").strip().lower() in ("1", "true", "yes")
    if (publication or pub_env or batch_invariant) and batch_size != 1:
        raise ConfigurationError(
            f"Publication mode requires batch_size=1 (got {batch_size}). "
            "Unset QREASON_PUBLICATION_MODE or use --batch-size 1."
        )


def setup_output_paths(
    cell_id: str,
    args_output: str | None,
    *,
    fresh: bool,
    suffix: str = "",
) -> tuple[Path, Path | None, Path | None]:
    """Resolve output path, archive root, and backup root."""
    if args_output:
        out_path = Path(args_output)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
    else:
        name = f"{cell_id}{suffix}.jsonl"
        out_path = REPO_ROOT / f"runs/raw/{name}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    archive_root = output_root_for(out_path)
    backup_root = (archive_root / "_backup") if archive_root else None

    if fresh and out_path.exists():
        print(f"--fresh: removing {out_path}")
        out_path.unlink()
        if archive_root:
            for sibling in (
                archive_root / "scored" / out_path.name,
                archive_root / "results" / f"{out_path.stem}_summary.json",
                archive_root / "checkpoints" / f"{out_path.stem}.json",
            ):
                if sibling.exists():
                    print(f"--fresh: removing {sibling}")
                    sibling.unlink()

    return out_path, archive_root, backup_root


def load_task_dataset(cell: dict[str, Any], limit: int | None):
    """Load task dataset with optional revision pin and limit."""
    task = cell["task"]
    print(f"Loading dataset: {task['dataset_id']} [{task['split']}]")
    if task.get("revision"):
        print(f"  revision pin: {task['revision']}")
    dataset = load_dataset_with_revision(task)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def guard_and_recover_resume(
    out_path: Path,
    cell: dict[str, Any],
    *,
    allow_resume: bool,
    backup_root: Path | None,
) -> list[dict[str, Any]]:
    """Resume guard + corrupt JSONL recovery; returns existing rows."""
    block_reason = resume_block_reason(out_path, cell, allow_resume=allow_resume)
    if block_reason:
        print(f"ERROR: {block_reason}", file=sys.stderr)
        sys.exit(1)

    ok, _ = validate_jsonl(out_path)
    if not ok:
        print(f"WARN: corrupt JSONL {out_path} — attempting restore from backup")
        if backup_root and recover_jsonl_from_backup(out_path, backup_root):
            print(f"Restored {out_path} from _backup/latest/raw/")
        else:
            corrupt = out_path.with_suffix(out_path.suffix + ".corrupt")
            out_path.replace(corrupt)
            print(f"Moved corrupt file → {corrupt}; starting fresh")

    return load_jsonl(out_path)
