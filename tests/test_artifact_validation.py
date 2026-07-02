"""Tests for experiment artifact homogeneity validation."""

import pytest

from src.runners.artifact_validation import (
    ArtifactValidationError,
    validate_experiment_homogeneity,
)


def test_homogeneous_rows_pass():
    rows = [
        {"cell_id": "a", "config_hash": "h1", "id": "1", "sample_index": 0},
        {"cell_id": "a", "config_hash": "h1", "id": "2", "sample_index": 0},
    ]
    validate_experiment_homogeneity(rows)


def test_mixed_cell_ids_fail():
    rows = [
        {"cell_id": "a", "config_hash": "h1", "id": "1", "sample_index": 0},
        {"cell_id": "b", "config_hash": "h2", "id": "2", "sample_index": 0},
    ]
    with pytest.raises(ArtifactValidationError):
        validate_experiment_homogeneity(rows)
