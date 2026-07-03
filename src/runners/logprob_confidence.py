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


def _logprob_for_token_id(item: Any, token_id: int) -> float | None:
    if not isinstance(item, dict):
        return _token_logprob(item)
    val = item.get(token_id)
    if val is None:
        val = item.get(str(token_id))
    if val is None:
        return None
    return _token_logprob(val)


def extract_token_logprobs(logprobs: Any, token_ids: list[int] | None = None) -> list[float]:
    """Normalize vLLM logprob payloads into per-generated-token logprobs.

    When token_ids is provided, take only the sampled token's entry at each step.
    With logprobs=1 and temperature>0, vLLM may return both top-1 and sampled token
    in the same dict; averaging all values inflates confidence.
    """
    if logprobs is None:
        return []
    if not isinstance(logprobs, list):
        return []
    values: list[float] = []
    if token_ids is not None and len(token_ids) == len(logprobs):
        for tid, item in zip(token_ids, logprobs):
            if item is None:
                continue
            lp = _logprob_for_token_id(item, int(tid))
            if lp is not None:
                values.append(lp)
        return values
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
    token_ids = getattr(choice, "token_ids", None)
    token_logprobs = extract_token_logprobs(logprobs, token_ids=token_ids)
    if not token_logprobs:
        return None
    mean_lp = sum(token_logprobs) / len(token_logprobs)
    confidence = min(1.0, max(0.0, math.exp(mean_lp)))
    return confidence, "normalized_sequence_logprob", mean_lp
