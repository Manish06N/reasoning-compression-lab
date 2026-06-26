"""Shared helpers for loading configs and building prompts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_model_path(model_cfg: Dict[str, Any], cell_cfg: Dict[str, Any]) -> str:
    if "model_path_override_env" in cell_cfg:
        env_name = cell_cfg["model_path_override_env"]
        default = cell_cfg.get("model_path_override_default", "")
        path = os.environ.get(env_name, default)
        if path:
            return str((REPO_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path))

    env_name = model_cfg.get("local_path_env", "")
    default = model_cfg.get("local_path_default", "")
    path = os.environ.get(env_name, default) if env_name else default
    if not path:
        return model_cfg["model_id"]
    return str((REPO_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path))


def build_prompt(template_file: str, question: str) -> str:
    template_path = REPO_ROOT / template_file
    template = template_path.read_text(encoding="utf-8")
    return template.format(question=question.strip())


def load_cell_config(cell_config_path: str | Path) -> Dict[str, Any]:
    cell = load_json(cell_config_path)
    cell["model"] = load_json(cell["model_config"])
    cell["task"] = load_json(cell["task_config"])
    cell["model_path"] = resolve_model_path(cell["model"], cell)
    return cell
