"""Calibration metrics: Brier, ECE, AUROC.

Adapted from Calibrating-LLMs-with-Consistency/source/calibrate/utils.py
(standalone — no external `calibration` package required).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np


def expected_calibration_error(
    confidences: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
) -> float:
    conf = np.asarray(confidences, dtype=float)
    lab = np.asarray(labels, dtype=int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        acc = lab[mask].mean()
        conf_mean = conf[mask].mean()
        ece += mask.mean() * abs(acc - conf_mean)
    return float(ece)


def brier_score(confidences: Sequence[float], labels: Sequence[int]) -> float:
    from sklearn.metrics import brier_score_loss

    return float(brier_score_loss(labels, confidences))


def auroc_score(confidences: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    from sklearn.metrics import roc_auc_score

    if len(set(labels)) < 2:
        return None
    try:
        return float(roc_auc_score(labels, confidences))
    except ValueError:
        return None


def aurc_score(confidences: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the risk-coverage curve (selective prediction)."""
    order = np.argsort(-np.asarray(confidences, dtype=float))
    sorted_labels = np.asarray(labels, dtype=int)[order]
    n = len(sorted_labels)
    risks = []
    coverages = []
    for k in range(1, n + 1):
        subset = sorted_labels[:k]
        risk = 1.0 - subset.mean()
        risks.append(risk)
        coverages.append(k / n)
    return float(np.trapz(risks, coverages))


def compute_calibration_metrics(
    confidences: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
) -> Dict[str, Optional[float]]:
    return {
        "brier": brier_score(confidences, labels),
        "ece": expected_calibration_error(confidences, labels, n_bins=n_bins),
        "auroc": auroc_score(confidences, labels),
        "aurc": aurc_score(confidences, labels),
    }
