"""Publication mode batch-size guard."""

import os
import pytest

from src.runners.inference_session import ConfigurationError, assert_publication_batch_size


def test_publication_mode_rejects_batch_gt_one():
    with pytest.raises(ConfigurationError):
        assert_publication_batch_size(4, publication=True)


def test_publication_env_rejects_batch_gt_one(monkeypatch):
    monkeypatch.setenv("QREASON_PUBLICATION_MODE", "1")
    with pytest.raises(ConfigurationError):
        assert_publication_batch_size(2, publication=False)


def test_normal_mode_allows_batch():
    assert_publication_batch_size(4, publication=False)
