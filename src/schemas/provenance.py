"""Run provenance and config hashing (V8.2 §11.1)."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.runners.config_hash import hash_material, stable_hash
from src.runners.config_utils import REPO_ROOT
from src.runners.revision_resolver import resolve_dataset_revision, resolve_model_revision


def git_commit_short(repo: Path | None = None) -> str:
    root = repo or REPO_ROOT
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def config_hash(
    cell: Mapping[str, Any],
    *,
    prompt_template_file: str | None = None,
    batch_size: int = 1,
    n_samples: int | None = None,
    max_model_len: int | None = None,
) -> str:
    task = cell.get("task") or {}
    tmpl = prompt_template_file or task.get("prompt_template_file", "")
    model = cell.get("model") or {}
    model_rev = resolve_model_revision(model, str(cell.get("model_path", "")))
    dataset_rev = resolve_dataset_revision(task)
    material = hash_material(
        cell,
        prompt_template_file=tmpl,
        batch_size=batch_size,
        n_samples=n_samples,
        max_model_len=max_model_len,
        model_revision=model_rev,
        dataset_revision=dataset_rev,
    )
    return stable_hash(material)


def input_text_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def make_run_id(cell_id: str, seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{cell_id}-s{seed}-{ts}"


def provenance_fields(
    cell: Mapping[str, Any],
    *,
    prompt_template_file: str,
    batch_size: int = 1,
    n_samples: int | None = None,
    max_model_len: int | None = None,
) -> dict[str, Any]:
    task = cell.get("task") or {}
    model = cell.get("model") or {}
    model_path = str(cell.get("model_path", ""))
    return {
        "run_id": make_run_id(str(cell["cell_id"]), int(cell["seed"])),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_short(),
        "config_hash": config_hash(
            cell,
            prompt_template_file=prompt_template_file,
            batch_size=batch_size,
            n_samples=n_samples,
            max_model_len=max_model_len,
        ),
        "prompt_template_version": Path(prompt_template_file).name,
        "prompt_template_file": prompt_template_file,
        "dataset_id": task.get("dataset_id"),
        "dataset_split": task.get("split"),
        "dataset_revision": resolve_dataset_revision(task),
        "model_revision": resolve_model_revision(model, model_path),
        "schema_version": "raw_response.v1",
    }
