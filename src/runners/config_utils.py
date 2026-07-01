"""Shared helpers for loading configs and building prompts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML required for decoding configs. pip install pyyaml") from exc
    return yaml.safe_load(text)


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


def build_prompt(template_file: str, **fields: str) -> str:
    template_path = REPO_ROOT / template_file
    template = template_path.read_text(encoding="utf-8")
    return template.format(**fields)


def load_decoding_from_file(decoding_file: str | Path) -> Dict[str, Any]:
    loaded = load_yaml(decoding_file)
    return {
        "temperature": loaded.get("temperature", 0.6),
        "top_p": loaded.get("top_p", 0.95),
        "max_tokens": loaded.get("max_tokens", 32768),
        "max_model_len": loaded.get("max_model_len"),
        "repetition_penalty": loaded.get("repetition_penalty"),
    }


def load_decoding(cell_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "decoding" in cell_cfg:
        return dict(cell_cfg["decoding"])
    decoding_file = cell_cfg.get("decoding_config")
    if decoding_file:
        return load_decoding_from_file(decoding_file)
    return {"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768}


def load_cell_config(cell_config_path: str | Path) -> Dict[str, Any]:
    cell = load_json(cell_config_path)
    cell["model"] = load_json(cell["model_config"])
    cell["task"] = load_json(cell["task_config"])
    cell["model_path"] = resolve_model_path(cell["model"], cell)
    cell["decoding"] = load_decoding(cell)
    if "seed" not in cell and cell.get("decoding_config"):
        dec = load_yaml(cell["decoding_config"])
        if "seed" in dec and "seed" not in cell:
            cell["seed"] = dec["seed"]
    return cell
