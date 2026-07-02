"""Immutable runtime specification for a single inference run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.runners.config_hash import hash_material, stable_hash
from src.runners.revision_resolver import resolve_dataset_revision, resolve_model_revision


@dataclass(frozen=True)
class RunSpec:
    """Single source of truth for run identity and config_hash material."""

    cell: Mapping[str, Any]
    prompt_template_file: str
    batch_size: int
    n_samples: int | None
    max_model_len: int | None
    publication_mode: bool


def run_spec_hash(spec: RunSpec) -> str:
    """Compute config_hash once from the full runtime spec."""
    model = spec.cell.get("model") or {}
    task = spec.cell.get("task") or {}
    material = hash_material(
        spec.cell,
        prompt_template_file=spec.prompt_template_file,
        batch_size=spec.batch_size,
        n_samples=spec.n_samples,
        max_model_len=spec.max_model_len,
        model_revision=resolve_model_revision(model, str(spec.cell.get("model_path", ""))),
        dataset_revision=resolve_dataset_revision(task),
    )
    return stable_hash(material)


def run_spec_from_cell(
    cell: Mapping[str, Any],
    *,
    prompt_template_file: str,
    batch_size: int = 1,
    n_samples: int | None = None,
    max_model_len: int | None = None,
    publication_mode: bool = False,
) -> RunSpec:
    return RunSpec(
        cell=cell,
        prompt_template_file=prompt_template_file,
        batch_size=batch_size,
        n_samples=n_samples,
        max_model_len=max_model_len,
        publication_mode=publication_mode,
    )
