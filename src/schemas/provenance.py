"""Run provenance and config hashing (V8.2 §11.1)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.runners.config_utils import REPO_ROOT


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


def stable_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def config_hash(cell: Mapping[str, Any]) -> str:
    material = {
        "cell_id": cell.get("cell_id"),
        "model_config": cell.get("model_config"),
        "task_config": cell.get("task_config"),
        "quant_config": cell.get("quant_config"),
        "seed": cell.get("seed"),
        "decoding": cell.get("decoding"),
        "prompt_profile": cell.get("prompt_profile"),
        "model_path": cell.get("model_path"),
    }
    return stable_hash(material)


def input_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def make_run_id(cell_id: str, seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{cell_id}-s{seed}-{ts}"


def provenance_fields(cell: Mapping[str, Any], *, prompt_template_file: str) -> dict[str, Any]:
    task = cell.get("task") or {}
    return {
        "run_id": make_run_id(str(cell["cell_id"]), int(cell["seed"])),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_short(),
        "config_hash": config_hash(cell),
        "prompt_template_version": Path(prompt_template_file).name,
        "prompt_template_file": prompt_template_file,
        "dataset_id": task.get("dataset_id"),
        "dataset_split": task.get("split"),
        "schema_version": "raw_response.v1",
    }
