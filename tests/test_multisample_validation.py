"""Tests for multisample group validation."""

import pytest

from src.runners.multisample_validation import (
    MultisampleValidationError,
    validate_multisample_groups,
)


def test_complete_maj5_group_valid():
    rows = [
        {"id": "1", "sample_index": i, "n_samples": 5}
        for i in range(5)
    ]
    report = validate_multisample_groups(rows, n_samples=5)
    assert report["valid"] is True


def test_incomplete_group_fails_in_publication_mode():
    rows = [{"id": "1", "sample_index": 0, "n_samples": 5}]
    with pytest.raises(MultisampleValidationError):
        validate_multisample_groups(rows, n_samples=5, publication_mode=True)
