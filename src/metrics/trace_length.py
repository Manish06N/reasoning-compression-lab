"""Reasoning trace length metrics.

Adapted from Quantized-Reasoning-Models/make_stats_table.py (get_num_steps, get_num_tokens).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def count_reasoning_steps(text: str, step_token: str = "\n\n") -> int:
    if not text:
        return 0
    return len(text.split(step_token))


def count_tokens(text: str, tokenizer=None) -> Optional[int]:
    if tokenizer is None:
        return None
    return len(tokenizer.encode(text))


def trace_stats(completion: str, tokenizer=None, step_token: str = "\n\n") -> Dict[str, Any]:
    return {
        "reasoning_steps": count_reasoning_steps(completion, step_token=step_token),
        "completion_tokens_est": count_tokens(completion, tokenizer),
    }


def summarize_trace_rows(rows: list[dict]) -> dict:
    steps = [r["reasoning_steps"] for r in rows if r.get("reasoning_steps") is not None]
    tokens = [r.get("completion_tokens") or r.get("completion_tokens_est") for r in rows]
    tokens = [t for t in tokens if t is not None]
    summary = {}
    if steps:
        summary["reasoning_steps_mean"] = sum(steps) / len(steps)
    if tokens:
        summary["completion_tokens_mean"] = sum(tokens) / len(tokens)
    return summary
