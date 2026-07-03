"""Tests for logprob confidence extraction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.runners.logprob_confidence import confidence_from_vllm_logprobs, extract_token_logprobs


def test_extract_token_logprobs_from_dict_list():
    logprobs = [{"42": SimpleNamespace(logprob=-0.5)}, {"43": SimpleNamespace(logprob=-1.0)}]
    assert extract_token_logprobs(logprobs) == [-0.5, -1.0]


def test_confidence_from_vllm_logprobs():
    choice = SimpleNamespace(
        token_ids=[1, 2],
        logprobs=[{"1": SimpleNamespace(logprob=-0.1)}, {"2": SimpleNamespace(logprob=-0.3)}],
    )
    confidence, source, mean_lp = confidence_from_vllm_logprobs(choice)
    assert source == "normalized_sequence_logprob"
    assert mean_lp == pytest.approx(-0.2)
    assert 0.0 < confidence <= 1.0


def test_extract_token_logprobs_uses_sampled_token_only():
    """Top-1 and sampled token in same dict must not double-count."""
    logprobs = [
        {"10": SimpleNamespace(logprob=-0.5), "99": SimpleNamespace(logprob=-0.1)},
        {"11": SimpleNamespace(logprob=-1.0)},
        {"12": SimpleNamespace(logprob=-2.0), "88": SimpleNamespace(logprob=-0.2)},
    ]
    token_ids = [99, 11, 12]
    assert extract_token_logprobs(logprobs, token_ids=token_ids) == [-0.1, -1.0, -2.0]
    inflated = extract_token_logprobs(logprobs)
    assert len(inflated) > len(token_ids)
