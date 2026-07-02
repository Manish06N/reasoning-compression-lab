"""Block unsafe resume into stale or bad-decoding archives."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from src.runners.checkpoint_utils import load_jsonl
from src.runners.config_utils import REPO_ROOT
from src.runners.publication_mode import code_changed_since
from src.runners.run_spec import RunSpec, run_spec_hash
from src.schemas.provenance import git_commit_short

# Archives known to contain pre-fix decoding (no repetition_penalty in rows).
FORBIDDEN_ARCHIVE_MARKERS = (
    "outputs-hpc-2a100-main-2026-06-29",
)

INVALID_FOR_PUBLICATION_MARKER = "INVALID_FOR_PUBLICATION.txt"
MARKER_WALK_MAX_DEPTH = 6


def _forbidden_path_patterns() -> tuple[str, ...]:
    extra = os.environ.get("QREASON_FORBIDDEN_ARCHIVE_PATTERNS", "").strip()
    if not extra:
        return FORBIDDEN_ARCHIVE_MARKERS
    from_env = tuple(part.strip() for part in extra.split(",") if part.strip())
    return FORBIDDEN_ARCHIVE_MARKERS + from_env


def _has_invalid_publication_marker(path: Path) -> bool:
    current = path.resolve()
    if current.is_file():
        current = current.parent
    for _ in range(MARKER_WALK_MAX_DEPTH):
        if (current / INVALID_FOR_PUBLICATION_MARKER).is_file():
            return True
        if current.parent == current:
            break
        current = current.parent
    return False


def archive_is_forbidden(output_path: Path) -> bool:
    if _has_invalid_publication_marker(output_path):
        return True
    text = str(output_path.resolve())
    return any(marker in text for marker in _forbidden_path_patterns())


def allow_resume_from_env() -> bool:
    return os.environ.get("QREASON_ALLOW_RESUME", "").strip().lower() in ("1", "true", "yes")


def allow_bad_archive_from_env() -> bool:
    return os.environ.get("QREASON_ALLOW_BAD_ARCHIVE", "").strip().lower() in ("1", "true", "yes")


def resume_block_reason(
    out_path: Path,
    cell: Mapping[str, Any],
    *,
    allow_resume: bool,
    run_spec: RunSpec | None = None,
) -> str | None:
    """Return error message if resume must be blocked, else None."""
    if allow_resume:
        return None
    if not out_path.exists():
        return None

    if archive_is_forbidden(out_path) and not allow_bad_archive_from_env():
        return (
            f"Refusing to use forbidden archive path: {out_path}. "
            f"Place {INVALID_FOR_PUBLICATION_MARKER} marks invalid trees; "
            "patterns also come from QREASON_FORBIDDEN_ARCHIVE_PATTERNS. "
            "Delete the old folder or set a new QREASON_OUTPUT_ROOT. "
            "Override only with QREASON_ALLOW_BAD_ARCHIVE=1 or QREASON_ALLOW_RESUME=1 (not recommended)."
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

    if run_spec is not None:
        expected_hash = run_spec_hash(run_spec)
    else:
        from src.schemas.provenance import config_hash

        expected_hash = config_hash(cell)
    row_hashes = {r.get("config_hash") for r in rows if r.get("config_hash")}
    if row_hashes and expected_hash not in row_hashes:
        return (
            f"Refusing to resume {out_path}: config_hash mismatch "
            f"(rows={sorted(row_hashes)}, current={expected_hash}). Use --fresh."
        )

    if row_commits and current_commit != "unknown":
        for row_commit in sorted(row_commits):
            if code_changed_since(REPO_ROOT, str(row_commit)):
                return (
                    f"Refusing to resume {out_path}: code changed since row git {row_commit} "
                    f"(current HEAD {current_commit}). Use --fresh after code sync."
                )

    return None
