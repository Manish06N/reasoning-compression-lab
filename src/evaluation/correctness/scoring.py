"""Correctness scoring — canonical re-export (V8.2 evaluation/correctness)."""

from src.metrics.scoring import (  # noqa: F401
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
    "maj_at_k",
    "majority_vote_answer",
    "score_gpqa_item",
    "score_gsm8k_item",
    "score_item",
    "score_math_item",
    "summarize_failure_rates",
    "summarize_scored_rows",
]
