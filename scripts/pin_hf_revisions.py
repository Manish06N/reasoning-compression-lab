#!/usr/bin/env python3
"""Resolve and pin HuggingFace model/dataset revisions to commit SHAs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runners.revision_resolver import is_immutable_revision, require_immutable_revision


def _repo_kind(path: Path) -> str:
    return "dataset" if "tasks" in path.parts else "model"


def _repo_id(data: dict, path: Path) -> str:
    if "dataset_id" in data:
        return str(data["dataset_id"])
    return str(data["model_id"])


def pin_file(path: Path, *, dry_run: bool = False) -> str | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    current = data.get("revision")
    if is_immutable_revision(str(current) if current else None):
        return None
    from huggingface_hub import HfApi

    repo_id = _repo_id(data, path)
    kind = _repo_kind(path)
    info = HfApi().repo_info(repo_id, repo_type=kind)
    sha = info.sha
    if dry_run:
        print(f"would pin {path}: {repo_id} -> {sha}")
        return sha
    data["revision"] = sha
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"pinned {path}: {sha}")
    return sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Fail if any pin is not a SHA")
    args = parser.parse_args()

    paths = sorted((ROOT / "configs/models").glob("*.json")) + sorted(
        (ROOT / "configs/tasks").glob("*.json")
    )
    if args.verify:
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            label = path.relative_to(ROOT).as_posix()
            require_immutable_revision(data.get("revision"), label=label)
        print(f"Verified {len(paths)} config revision pins.")
        return

    for path in paths:
        pin_file(path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
