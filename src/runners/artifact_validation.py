"""Validate experiment artifact homogeneity before scoring."""

from __future__ import annotations

from typing import Any, Iterable, Sequence


class ArtifactValidationError(ValueError):
    """Raised when raw/scored rows fail homogeneity or completeness checks."""


def assert_single_value(rows: Sequence[dict[str, Any]], field: str) -> Any:
    values = {row.get(field) for row in rows if row.get(field) is not None}
    if len(values) > 1:
        raise ArtifactValidationError(
            f"Mixed {field} values in artifact: {sorted(values)!r}"
        )
    return next(iter(values)) if values else None


def assert_unique(rows: Sequence[dict[str, Any]], keys: Iterable[str]) -> None:
    key_tuple = tuple(keys)
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        identity = tuple(row.get(k) for k in key_tuple)
        if identity in seen:
            raise ArtifactValidationError(f"Duplicate row identity {identity!r}")
        seen.add(identity)


def assert_expected_row_count(
    rows: Sequence[dict[str, Any]],
    expected: int,
    *,
    label: str = "rows",
) -> None:
    if len(rows) != expected:
        raise ArtifactValidationError(
            f"Expected {expected} {label}, got {len(rows)}"
        )


def validate_experiment_homogeneity(
    rows: Sequence[dict[str, Any]],
    *,
    expected_row_count: int | None = None,
) -> None:
    """Ensure all rows belong to one experiment."""
    if not rows:
        raise ArtifactValidationError("No rows to validate")
    for field in (
        "cell_id",
        "config_hash",
        "task",
        "seed",
        "quant_config",
        "dataset_revision",
        "model_revision",
    ):
        assert_single_value(rows, field)
    assert_unique(rows, ("id", "sample_index"))
    if expected_row_count is not None:
        assert_expected_row_count(rows, expected_row_count)


def validate_experiment_homogeneity_warn(
    rows: Sequence[dict[str, Any]],
) -> list[str]:
    """Return warning messages without raising."""
    warnings: list[str] = []
    try:
        validate_experiment_homogeneity(rows)
    except ArtifactValidationError as exc:
        warnings.append(str(exc))
    return warnings
