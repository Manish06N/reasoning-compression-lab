"""Block unsafe resume into stale or bad-decoding archives."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from src.runners.checkpoint_utils import load_jsonl
from src.schemas.provenance import config_hash, git_commit_short

# Archives known to contain pre-fix decoding (no repetition_penalty in rows).
FORBIDDEN_ARCHIVE_MARKERS = (
    "outputs-hpc-2a100-main-2026-06-29",
)


def archive_is_forbidden(output_path: Path) -> bool:
    text = str(output_path.resolve())
    return any(marker in text for marker in FORBIDDEN_ARCHIVE_MARKERS)


def resume_block_reason(
    out_path: Path,
    cell: Mapping[str, Any],
    *,
    allow_resume: bool,
) -> str | None:
    """Return error message if resume must be blocked, else None."""
    if allow_resume:
        return None
    if not out_path.exists():
        return None

    if archive_is_forbidden(out_path):
        return (
            f"Refusing to use forbidden archive path: {out_path}. "
            "Delete the old folder or set a new QREASON_OUTPUT_ROOT. "
            "Override only with QREASON_ALLOW_RESUME=1 (not recommended)."
        )

    rows = load_jsonl(out_path)
    if not rows:
        return None

    expected_penalty = (cell.get("decoding") or {}).get("repetition_penalty")
    if expected_penalty is not None:
        without = sum(1 for r in rows if r.get("decoding_repetition_penalty") is None)
        if without == len(rows):
            return (
                f"Refusing to resume {out_path}: {len(rows)} existing rows lack "
                f"decoding_repetition_penalty (pre-fix run). "
                "Use --fresh or delete the raw JSONL."
            )

    current_commit = git_commit_short()
    row_commits = {r.get("git_commit") for r in rows if r.get("git_commit")}
    if row_commits and current_commit != "unknown" and current_commit not in row_commits:
        return (
            f"Refusing to resume {out_path}: rows from git {sorted(row_commits)}, "
            f"current HEAD {current_commit}. Use --fresh after code sync."
        )

    expected_hash = config_hash(cell)
    row_hashes = {r.get("config_hash") for r in rows if r.get("config_hash")}
    if row_hashes and expected_hash not in row_hashes:
        return (
            f"Refusing to resume {out_path}: config_hash mismatch "
            f"(rows={sorted(row_hashes)}, current={expected_hash}). Use --fresh."
        )

    return None


def allow_resume_from_env() -> bool:
    return os.environ.get("QREASON_ALLOW_RESUME", "").strip() in ("1", "true", "yes")
