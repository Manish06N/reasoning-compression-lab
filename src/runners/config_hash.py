"""Content-based configuration hashing (machine-independent)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.runners.config_utils import REPO_ROOT


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _resolve_repo_path(rel_or_abs: str | Path) -> Path:
    path = Path(rel_or_abs)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def config_file_hashes(cell: Mapping[str, Any]) -> dict[str, str]:
    """SHA256 (truncated) of referenced config file contents."""
    hashes: dict[str, str] = {}
    for key in ("model_config", "task_config"):
        ref = cell.get(key)
        if ref:
            path = _resolve_repo_path(ref)
            if path.exists():
                hashes[key] = _sha256_file(path)
    decoding_file = cell.get("decoding_config")
    if decoding_file:
        path = _resolve_repo_path(decoding_file)
        if path.exists():
            hashes["decoding_config"] = _sha256_file(path)
    return hashes


def prompt_content_hash(prompt_template_file: str) -> str:
    path = _resolve_repo_path(prompt_template_file)
    if not path.exists():
        return ""
    return _sha256_file(path)


def hash_material(
    cell: Mapping[str, Any],
    *,
    prompt_template_file: str,
    batch_size: int = 1,
    n_samples: int | None = None,
    max_model_len: int | None = None,
    model_revision: str | None = None,
    dataset_revision: str | None = None,
) -> dict[str, Any]:
    """Build stable dict for config_hash (no absolute model_path)."""
    model = cell.get("model") or {}
    task = cell.get("task") or {}
    material: dict[str, Any] = {
        "cell_id": cell.get("cell_id"),
        "quant_config": cell.get("quant_config"),
        "seed": cell.get("seed"),
        "decoding": cell.get("decoding"),
        "prompt_profile": cell.get("prompt_profile"),
        "model_id": model.get("model_id"),
        "model_revision": model_revision or model.get("revision"),
        "dataset_id": task.get("dataset_id"),
        "dataset_config": task.get("config_name"),
        "dataset_revision": dataset_revision or task.get("revision"),
        "dataset_split": task.get("split"),
        "prompt_template_file": prompt_template_file,
        "prompt_content_hash": prompt_content_hash(prompt_template_file),
        "config_file_hashes": config_file_hashes(cell),
        "batch_size": batch_size,
    }
    if n_samples is not None:
        material["n_samples"] = n_samples
    if max_model_len is not None:
        material["max_model_len"] = max_model_len
    elif model.get("max_model_len") is not None:
        material["max_model_len"] = model.get("max_model_len")
    return material


def stable_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
