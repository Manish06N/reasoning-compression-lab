"""Validate complete maj@k multisample groups."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence


class MultisampleValidationError(ValueError):
    """Raised when maj@k rows are incomplete or inconsistent."""


def validate_multisample_groups(
    rows: Sequence[dict[str, Any]],
    *,
    n_samples: int,
    publication_mode: bool = False,
) -> dict[str, Any]:
    """Ensure every item has exactly n_samples with unique indices."""
    by_item: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_item[row.get("id")].append(row)

    incomplete: list[Any] = []
    duplicate_indices: list[Any] = []
    for item_id, group in by_item.items():
        indices = [r.get("sample_index") for r in group]
        if len(set(indices)) != len(indices):
            duplicate_indices.append(item_id)
        if len(group) != n_samples:
            incomplete.append(item_id)

    report = {
        "n_items": len(by_item),
        "expected_samples_per_item": n_samples,
        "incomplete_items": incomplete,
        "duplicate_index_items": duplicate_indices,
        "valid": not incomplete and not duplicate_indices,
    }
    if publication_mode and not report["valid"]:
        raise MultisampleValidationError(
            f"Incomplete maj@{n_samples} groups: incomplete={len(incomplete)}, "
            f"duplicate_indices={len(duplicate_indices)}"
        )
    return report


def infer_n_samples(rows: Sequence[dict[str, Any]]) -> int | None:
    values = [row.get("n_samples") for row in rows if row.get("n_samples") is not None]
    if not values:
        return None
    counts = Counter(values)
    return counts.most_common(1)[0][0]
