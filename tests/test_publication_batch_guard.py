"""Publication mode batch-size guard."""

from pathlib import Path

import pytest

from src.runners.inference_session import ConfigurationError, assert_publication_batch_size

ROOT = Path(__file__).resolve().parents[1]


def test_publication_mode_rejects_batch_gt_one():
    with pytest.raises(ConfigurationError):
        assert_publication_batch_size(4, publication=True)


def test_publication_env_rejects_batch_gt_one(monkeypatch):
    monkeypatch.setenv("QREASON_PUBLICATION_MODE", "1")
    with pytest.raises(ConfigurationError):
        assert_publication_batch_size(2, publication=False)


def test_hpc_launcher_exports_publication_mode():
    launcher = ROOT / "scripts/hpc/run_hpc_2a100_publication.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "QREASON_PUBLICATION_MODE=1" in text
    assert "VLLM_BATCH_INVARIANT=1" in text
    assert "--publication" in text


def test_normal_mode_allows_batch():
    assert_publication_batch_size(4, publication=False)
