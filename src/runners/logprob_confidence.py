"""Derive calibration confidence from vLLM token logprobs."""

from __future__ import annotations

import math
from typing import Any


def _token_logprob(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    lp = getattr(value, "logprob", None)
    if lp is not None:
        return float(lp)
    if isinstance(value, dict):
        if "logprob" in value:
            return float(value["logprob"])
        if len(value) == 1:
            return _token_logprob(next(iter(value.values())))
    return None


def extract_token_logprobs(logprobs: Any) -> list[float]:
    """Normalize vLLM logprob payloads into a flat list of token logprobs."""
    if logprobs is None:
        return []
    values: list[float] = []
    if isinstance(logprobs, list):
        for item in logprobs:
            if item is None:
                continue
            if isinstance(item, dict):
                for val in item.values():
                    lp = _token_logprob(val)
                    if lp is not None:
                        values.append(lp)
            else:
                lp = _token_logprob(item)
                if lp is not None:
                    values.append(lp)
    return values


def confidence_from_vllm_logprobs(choice: Any) -> tuple[float, str, float] | None:
    """Return (confidence, source, mean_token_logprob) from a vLLM completion choice."""
    logprobs = getattr(choice, "logprobs", None)
    token_logprobs = extract_token_logprobs(logprobs)
    if not token_logprobs:
        return None
    mean_lp = sum(token_logprobs) / len(token_logprobs)
    confidence = min(1.0, max(0.0, math.exp(mean_lp)))
    return confidence, "normalized_sequence_logprob", mean_lp
