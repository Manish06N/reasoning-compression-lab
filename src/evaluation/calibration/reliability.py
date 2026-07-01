"""Adaptive ECE, NLL, and reliability diagram bins (V8.2 §6.5)."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence


def negative_log_likelihood(confidences: Sequence[float], labels: Sequence[int]) -> float:
    eps = 1e-12
    total = 0.0
    for conf, label in zip(confidences, labels):
        p = max(eps, min(1.0 - eps, float(conf)))
        total += -(label * math.log(p) + (1 - label) * math.log(1 - p))
    return total / len(labels) if labels else 0.0


def adaptive_ece(
    confidences: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 15,
) -> float:
    """Equal-mass binning ECE (adaptive bins by sample count)."""
    if not confidences:
        return 0.0
    pairs = sorted(zip(confidences, labels), key=lambda x: x[0])
    n = len(pairs)
    bin_size = max(1, n // n_bins)
    ece = 0.0
    for start in range(0, n, bin_size):
        chunk = pairs[start : start + bin_size]
        if not chunk:
            continue
        confs = [c for c, _ in chunk]
        labs = [l for _, l in chunk]
        acc = sum(labs) / len(labs)
        conf_mean = sum(confs) / len(confs)
        weight = len(chunk) / n
        ece += weight * abs(acc - conf_mean)
    return float(ece)


def reliability_diagram_bins(
    confidences: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    """Fixed-width bins for plotting reliability diagrams."""
    bins: List[Dict[str, float]] = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        mask = [(lo < c <= hi) if i > 0 else (lo <= c <= hi) for c in confidences]
        idxs = [j for j, m in enumerate(mask) if m]
        if not idxs:
            bins.append({"bin_lo": lo, "bin_hi": hi, "count": 0, "accuracy": 0.0, "confidence": 0.0})
            continue
        acc = sum(labels[j] for j in idxs) / len(idxs)
        conf = sum(confidences[j] for j in idxs) / len(idxs)
        bins.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "count": len(idxs),
                "accuracy": acc,
                "confidence": conf,
            }
        )
    return bins
