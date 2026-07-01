"""Paired McNemar test for binary correctness (same items, two configs)."""

from __future__ import annotations

import math
from typing import Sequence


def mcnemar_test(
    baseline_correct: Sequence[bool],
    variant_correct: Sequence[bool],
) -> dict[str, float | int | None]:
    """
    Compare paired binary outcomes on the same items.

    Returns counts b (baseline correct, variant wrong) and c (baseline wrong, variant correct),
    chi-square statistic with continuity correction, and two-sided p-value.
    """
    if len(baseline_correct) != len(variant_correct):
        raise ValueError("baseline and variant must have same length")
    b = c = 0
    for base, var in zip(baseline_correct, variant_correct):
        if base and not var:
            b += 1
        elif not base and var:
            c += 1
    n_discordant = b + c
    if n_discordant == 0:
        return {
            "b": b,
            "c": c,
            "n_discordant": 0,
            "statistic": 0.0,
            "p_value": 1.0,
            "effect_rate_diff": 0.0,
        }
    stat = (abs(b - c) - 1) ** 2 / n_discordant
    p_value = math.erfc(math.sqrt(stat / 2.0))
    n = len(baseline_correct)
    effect = (sum(variant_correct) - sum(baseline_correct)) / n if n else 0.0
    return {
        "b": b,
        "c": c,
        "n_discordant": n_discordant,
        "statistic": stat,
        "p_value": p_value,
        "effect_rate_diff": effect,
    }
