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
        logprobs=[{"1": SimpleNamespace(logprob=-0.1)}, {"2": SimpleNamespace(logprob=-0.3)}]
    )
    confidence, source, mean_lp = confidence_from_vllm_logprobs(choice)
    assert source == "normalized_sequence_logprob"
    assert mean_lp == pytest.approx(-0.2)
    assert 0.0 < confidence <= 1.0
