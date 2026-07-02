"""Paired McNemar test for binary correctness (same items, two configs)."""

from __future__ import annotations

import math
from typing import Sequence


def mcnemar_test(
    baseline_correct: Sequence[bool],
    variant_correct: Sequence[bool],
) -> dict[str, float | int | str | None]:
    """
    Compare paired binary outcomes on the same items.

    Uses exact binomial test for small discordant counts; asymptotic with
    continuity correction for larger samples.
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
    n = len(baseline_correct)
    effect = (sum(variant_correct) - sum(baseline_correct)) / n if n else 0.0
    if n_discordant == 0:
        return {
            "test": "mcnemar_exact",
            "b": b,
            "c": c,
            "n_discordant": 0,
            "statistic": 0.0,
            "p_value": 1.0,
            "effect_rate_diff": effect,
        }

    if n_discordant <= 25:
        from scipy.stats import binomtest

        p_value = float(
            binomtest(min(b, c), n=n_discordant, p=0.5, alternative="two-sided").pvalue
        )
        return {
            "test": "mcnemar_exact",
            "b": b,
            "c": c,
            "n_discordant": n_discordant,
            "statistic": None,
            "p_value": p_value,
            "effect_rate_diff": effect,
        }

    stat = max(abs(b - c) - 1, 0) ** 2 / n_discordant
    p_value = math.erfc(math.sqrt(stat / 2.0))
    return {
        "test": "mcnemar_asymptotic",
        "b": b,
        "c": c,
        "n_discordant": n_discordant,
        "statistic": stat,
        "p_value": p_value,
        "effect_rate_diff": effect,
    }
