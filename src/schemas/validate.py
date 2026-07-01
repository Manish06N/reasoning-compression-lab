"""JSON Schema validation for pipeline artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.runners.config_utils import REPO_ROOT

_SCHEMA_CACHE: dict[str, dict[str, Any]] = {}


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


def _validate_required_only(row: dict[str, Any], schema_name: str) -> list[str]:
    schema = load_schema(schema_name)
    required = schema.get("required", [])
    missing = [f for f in required if f not in row or row[f] is None]
    if missing:
        return [f"Missing required fields: {', '.join(missing)}"]
    return []


def validate_jsonl_rows(path: Path, schema_name: str = "raw_response.v1.json", *, limit: int | None = 10) -> dict[str, Any]:
    from src.runners.checkpoint_utils import load_jsonl

    rows = load_jsonl(path)
    if limit is not None:
        sample = rows[:limit]
    else:
        sample = rows
    errors: list[str] = []
    for i, row in enumerate(sample):
        row_errors = validate_row(row, schema_name)
        for err in row_errors:
            errors.append(f"row {i}: {err}")
    return {
        "path": str(path),
        "schema": schema_name,
        "rows_checked": len(sample),
        "total_rows": len(rows),
        "valid": len(errors) == 0,
        "errors": errors[:20],
    }
