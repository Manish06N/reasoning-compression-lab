"""Deprecated — import from src.evaluation.correctness.scoring instead."""

from __future__ import annotations

from src.evaluation.correctness.scoring import (  # noqa: F401
    bootstrap_ci,
    maj_at_k,
    majority_vote_answer,
    score_gpqa_item,
    score_gsm8k_item,
    score_item,
    score_math_item,
    summarize_failure_rates,
    summarize_scored_rows,
)

__all__ = [
    "bootstrap_ci",
    "maj_at_k",
    "majority_vote_answer",
    "score_gpqa_item",
    "score_gsm8k_item",
    "score_item",
    "score_math_item",
    "summarize_failure_rates",
    "summarize_scored_rows",
]
