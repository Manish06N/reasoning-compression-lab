"""vLLM LLM() kwargs aligned with QRM inference.py + configs/serving/vllm.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVING_PATH = ROOT / "configs" / "serving" / "vllm.yaml"

# QRM inference.py VLLMModelConfig defaults (COLM 2025 reproduction stack).
QRM_SERVING_DEFAULTS: Dict[str, Any] = {
    "gpu_memory_utilization": 0.9,
    "enable_prefix_caching": False,
    "enable_chunked_prefill": False,
    "enforce_eager": True,
}


def load_serving_defaults(path: Optional[Path] = None) -> Dict[str, Any]:
    cfg_path = path or DEFAULT_SERVING_PATH
    if not cfg_path.is_file():
        return dict(QRM_SERVING_DEFAULTS)
    with cfg_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    merged = dict(QRM_SERVING_DEFAULTS)
    for key in QRM_SERVING_DEFAULTS:
        if key in data and data[key] is not None:
            merged[key] = data[key]
    return merged


def build_llm_init_kwargs(
    model_path: str,
    model_cfg: Dict[str, Any],
    *,
    seed: Optional[int] = None,
    serving_defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build kwargs for vLLM LLM() with QRM serving parity fields."""
    defaults = serving_defaults or load_serving_defaults()
    llm_kwargs: Dict[str, Any] = {
        "model": model_path,
        "dtype": model_cfg.get("dtype", "bfloat16"),
        "max_model_len": model_cfg.get("max_model_len", 40960),
        "tensor_parallel_size": model_cfg.get("tensor_parallel_size", 1),
        "trust_remote_code": model_cfg.get("trust_remote_code", True),
    }

    for key in (
        "enforce_eager",
        "gpu_memory_utilization",
        "enable_prefix_caching",
        "enable_chunked_prefill",
    ):
        if model_cfg.get(key) is not None:
            llm_kwargs[key] = model_cfg[key]
        elif defaults.get(key) is not None:
            llm_kwargs[key] = defaults[key]

    if model_cfg.get("quantization"):
        llm_kwargs["quantization"] = model_cfg["quantization"]
    if model_cfg.get("kv_cache_dtype"):
        llm_kwargs["kv_cache_dtype"] = model_cfg["kv_cache_dtype"]

    engine_seed = seed if seed is not None else model_cfg.get("seed")
    if engine_seed is not None:
        llm_kwargs["seed"] = int(engine_seed)

    return llm_kwargs