"""Decode/sampling parameter helpers (sober-reasoning + vLLM SamplingParams alignment)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Required keys that must reach vLLM SamplingParams (see sober-reasoning main.py GenerationParameters).
SAMPLING_PARAM_KEYS = ("temperature", "top_p", "max_tokens", "seed", "repetition_penalty")

# sober-reasoning passes seed to both model config and generation_parameters.
SOBER_REFERENCE = {
    "temperature": 0.8,
    "top_p": 0.9,
    "max_new_tokens": 32768,
    "repetition_penalty": None,  # optional CLI flag
    "seed_in_generation_parameters": True,
}

# QRM reproduction gate (inference.py defaults); repetition_penalty added for vLLM 0.8.x anti-loop.
QRM_REPRO_REFERENCE = {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_new_tokens": 32768,
    "seed": 42,
    "repetition_penalty": None,
}


def build_sampling_params_dict(decoding: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Build the dict passed to vLLM SamplingParams (testable without GPU)."""
    params: Dict[str, Any] = {
        "temperature": decoding.get("temperature", 0.6),
        "top_p": decoding.get("top_p", 0.95),
        "max_tokens": decoding.get("max_tokens", 32768),
        "seed": seed,
    }
    if decoding.get("repetition_penalty") is not None:
        params["repetition_penalty"] = decoding["repetition_penalty"]
    if decoding.get("frequency_penalty") is not None:
        params["frequency_penalty"] = decoding["frequency_penalty"]
    return params


def sample_seed_for_draw(base_seed: int, sample_index: int) -> int:
    """Distinct vLLM seed per maj@k draw (Calibrating-LLMs / Level B pilot)."""
    return int(base_seed) * 1000 + int(sample_index)


def verify_decoding_for_vllm(
    decoding: Dict[str, Any],
    cell_seed: int,
    *,
    require_repetition_penalty: bool = True,
) -> Tuple[bool, List[str]]:
    """Return (ok, messages) after checking repro_qrm.yaml → SamplingParams wiring."""
    messages: List[str] = []
    params = build_sampling_params_dict(decoding, cell_seed)

    if params.get("seed") != cell_seed:
        messages.append(f"FAIL: SamplingParams seed={params.get('seed')} != cell seed={cell_seed}")
    else:
        messages.append(f"OK: seed={cell_seed} forwarded to SamplingParams")

    for key in ("temperature", "top_p", "max_tokens"):
        if key not in params:
            messages.append(f"FAIL: missing SamplingParams key {key}")
        else:
            messages.append(f"OK: {key}={params[key]}")

    rp = decoding.get("repetition_penalty")
    if require_repetition_penalty:
        if rp is None:
            messages.append("WARN: repetition_penalty not set (R1 loops may hit 32k cap on vLLM 0.8.x)")
        elif "repetition_penalty" not in params:
            messages.append("FAIL: repetition_penalty in YAML but not in SamplingParams")
        else:
            messages.append(f"OK: repetition_penalty={rp} (sober-reasoning optional flag pattern)")

    ok = not any(m.startswith("FAIL:") for m in messages)
    return ok, messages
