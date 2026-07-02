"""JSON Schema validation for pipeline artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.runners.config_utils import REPO_ROOT

_SCHEMA_CACHE: dict[str, dict[str, Any]] = {}


class SchemaValidationError(ValueError):
    """Raised when a row fails schema validation."""


def load_schema(name: str) -> dict[str, Any]:
    if name in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[name]
    path = REPO_ROOT / "schemas" / name
    schema = json.loads(path.read_text(encoding="utf-8"))
    _SCHEMA_CACHE[name] = schema
    return schema


def validate_row(row: dict[str, Any], schema_name: str = "raw_response.v1.json") -> list[str]:
    """Return list of validation errors (empty if valid)."""
    try:
        import jsonschema
    except ImportError:
        return _validate_required_only(row, schema_name)

    schema = load_schema(schema_name)
    validator = jsonschema.Draft7Validator(schema)
    return [e.message for e in sorted(validator.iter_errors(row), key=lambda e: e.path)]


def validate_row_or_raise(row: dict[str, Any], schema_name: str = "raw_response.v1.json") -> None:
    errors = validate_row(row, schema_name)
    if errors:
        raise SchemaValidationError("; ".join(errors[:5]))


def _validate_required_only(row: dict[str, Any], schema_name: str) -> list[str]:
    schema = load_schema(schema_name)
    required = schema.get("required", [])
    missing = [f for f in required if f not in row or row[f] is None]
    if missing:
        return [f"Missing required fields: {', '.join(missing)}"]
    return []


def _sample_indices(total: int, *, every_nth: int = 100) -> list[int]:
    if total == 0:
        return []
    indices = {0, total - 1}
    for i in range(0, total, every_nth):
        indices.add(i)
    return sorted(indices)


def validate_jsonl_rows(
    path: Path,
    schema_name: str = "raw_response.v1.json",
    *,
    limit: int | None = None,
    every_nth: int = 100,
) -> dict[str, Any]:
    from src.runners.checkpoint_utils import load_jsonl

    rows = load_jsonl(path)
    if limit is not None:
        sample_idx = list(range(min(limit, len(rows))))
    else:
        sample_idx = _sample_indices(len(rows), every_nth=every_nth)
    errors: list[str] = []
    for i in sample_idx:
        row_errors = validate_row(rows[i], schema_name)
        for err in row_errors:
            errors.append(f"row {i}: {err}")
    return {
        "path": str(path),
        "schema": schema_name,
        "rows_checked": len(sample_idx),
        "total_rows": len(rows),
        "valid": len(errors) == 0,
        "errors": errors[:20],
    }


def validate_jsonl_sample(path: Path, schema_name: str = "raw_response.v1.json") -> dict[str, Any]:
    """Validate first, last, and periodic rows (default for score_run preflight)."""
    return validate_jsonl_rows(path, schema_name, limit=None, every_nth=100)
