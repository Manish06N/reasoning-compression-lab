"""Publication vs debug execution mode helpers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

PUBLICATION_CODE_PATHS: tuple[str, ...] = (
    "src",
    "scripts",
    "configs",
    "prompts",
    "schemas",
    "papers",
    "slurm",
    "tests",
    "pyproject.toml",
)


def is_publication_mode(*, cli_flag: bool = False) -> bool:
    """True when publication safeguards must fail closed."""
    if cli_flag:
        return True
    env = os.environ.get("QREASON_PUBLICATION_MODE", "").strip().lower()
    return env in ("1", "true", "yes")


def _git_diff_quiet(repo_root: Path, *extra_args: str) -> None:
    args = ["git", "diff", "--quiet", *extra_args, "--", *PUBLICATION_CODE_PATHS]
    subprocess.run(args, cwd=repo_root, check=True, capture_output=True)


def assert_code_paths_clean(repo_root) -> None:
    """Refuse publication runs when code/config paths are dirty."""
    root = Path(repo_root)
    try:
        _git_diff_quiet(root)
        _git_diff_quiet(root, "--cached")
    except FileNotFoundError as exc:
        raise SystemExit(
            "ERROR: Publication run requires Git installed and a git checkout."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "ERROR: Publication run requires a clean git working tree for code paths "
            f"({', '.join(PUBLICATION_CODE_PATHS)}). "
            "Commit or stash code changes, or use a fresh checkout."
        ) from exc


def code_changed_since(repo_root, commit: str) -> bool:
    """True when code paths differ between commit and HEAD."""
    if not commit or commit == "unknown":
        return False
    root = Path(repo_root)
    args = ["git", "diff", "--quiet", f"{commit}..HEAD", "--", *PUBLICATION_CODE_PATHS]
    try:
        subprocess.run(args, cwd=root, check=True, capture_output=True)
        return False
    except subprocess.CalledProcessError:
        return True
    except FileNotFoundError:
        return False


def assert_clean_git_tree(repo_root) -> None:
    """Refuse publication runs on dirty code paths (output bookkeeping allowed)."""
    assert_code_paths_clean(repo_root)
