"""Holm step-down correction for planned pairwise comparisons."""

from __future__ import annotations

from typing import Sequence


def holm_adjusted_pvalues(p_values: Sequence[float]) -> list[float]:
    """Return Holm-adjusted p-values in the same order as input."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [1.0] * m
    prev = 0.0
    for rank, (idx, p) in enumerate(indexed, start=1):
        adj = min(1.0, (m - rank + 1) * p)
        adj = max(adj, prev)
        adjusted[idx] = adj
        prev = adj
    return adjusted
