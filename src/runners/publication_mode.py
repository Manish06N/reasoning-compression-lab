"""Publication vs debug execution mode helpers."""

from __future__ import annotations

import os


def is_publication_mode(*, cli_flag: bool = False) -> bool:
    """True when publication safeguards must fail closed."""
    if cli_flag:
        return True
    env = os.environ.get("QREASON_PUBLICATION_MODE", "").strip().lower()
    return env in ("1", "true", "yes")


def assert_clean_git_tree(repo_root) -> None:
    """Refuse publication runs on dirty working trees."""
    import subprocess
    from pathlib import Path

    root = Path(repo_root)
    for args in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        try:
            subprocess.run(args, cwd=root, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                "ERROR: Publication run requires a clean git working tree. "
                "Commit or stash changes, or use a fresh checkout."
            ) from exc
