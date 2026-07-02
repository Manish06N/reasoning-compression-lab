"""Answer consistency metrics with semantic equivalence."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from src.evaluation.correctness.scoring import answers_semantically_equivalent
from src.extraction.math_extractor import normalize_answer


def agreement_rates(answers: Sequence[str]) -> dict[str, Any]:
    """Compute raw, normalized, and semantic agreement for one item's samples."""
    if not answers:
        return {
            "raw_string_agreement": 0.0,
            "normalized_string_agreement": 0.0,
            "semantic_equivalence_agreement": 0.0,
            "confidence_method": "self_consistency_5",
        }
    raw_counts = Counter(str(a) for a in answers)
    raw_agreement = raw_counts.most_common(1)[0][1] / len(answers)

    normalized = [normalize_answer(a) or "" for a in answers]
    norm_counts = Counter(normalized)
    norm_agreement = norm_counts.most_common(1)[0][1] / len(normalized)

    clusters: list[list[str]] = []
    for ans in answers:
        placed = False
        for cluster in clusters:
            if answers_semantically_equivalent(cluster[0], ans):
                cluster.append(ans)
                placed = True
                break
        if not placed:
            clusters.append([ans])
    largest_cluster = max(len(c) for c in clusters)
    semantic_agreement = largest_cluster / len(answers)

    return {
        "raw_string_agreement": raw_agreement,
        "normalized_string_agreement": norm_agreement,
        "semantic_equivalence_agreement": semantic_agreement,
        "confidence_method": "self_consistency_5",
    }
