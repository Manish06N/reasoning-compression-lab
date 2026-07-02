"""Resolve pinned or runtime model/dataset revisions for provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def resolve_dataset_revision(task_cfg: Mapping[str, Any]) -> str | None:
    """Return pinned revision or None (caller may resolve at load time)."""
    pinned = task_cfg.get("revision")
    if pinned:
        return str(pinned)
    return None


def resolve_model_revision(model_cfg: Mapping[str, Any], model_path: str) -> str | None:
    """Read revision from config pin, local config.json, or Hub id."""
    pinned = model_cfg.get("revision")
    if pinned:
        return str(pinned)
    path = Path(model_path)
    if path.is_dir():
        config_json = path / "config.json"
        if config_json.exists():
            try:
                data = json.loads(config_json.read_text(encoding="utf-8"))
                for key in ("_commit_hash", "commit_hash", "revision"):
                    if data.get(key):
                        return str(data[key])
            except (json.JSONDecodeError, OSError):
                pass
    model_id = model_cfg.get("model_id")
    if model_id and not path.is_dir():
        return str(model_id)
    return None


def load_dataset_with_revision(task_cfg: Mapping[str, Any], *, split: str | None = None):
    """Load HuggingFace dataset using pinned revision when available."""
    from datasets import load_dataset

    dataset_id = task_cfg["dataset_id"]
    config_name = task_cfg.get("config_name")
    split_name = split or task_cfg["split"]
    revision = resolve_dataset_revision(task_cfg)
    kwargs: dict[str, Any] = {"split": split_name}
    if revision:
        kwargs["revision"] = revision
    if config_name:
        return load_dataset(dataset_id, config_name, **kwargs)
    return load_dataset(dataset_id, **kwargs)
