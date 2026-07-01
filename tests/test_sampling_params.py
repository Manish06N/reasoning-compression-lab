"""Tests for SamplingParams wiring (sober-reasoning / vLLM alignment)."""

from src.runners.sampling_utils import (
    build_sampling_params_dict,
    sample_seed_for_draw,
    verify_decoding_for_vllm,
)


def test_build_sampling_params_includes_repetition_penalty():
    decoding = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 32768,
        "repetition_penalty": 1.05,
    }
    params = build_sampling_params_dict(decoding, seed=0)
    assert params["seed"] == 0
    assert params["repetition_penalty"] == 1.05
    assert params["temperature"] == 0.6


def test_verify_decoding_ok_with_repetition_penalty():
    decoding = {"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768, "repetition_penalty": 1.05}
    ok, messages = verify_decoding_for_vllm(decoding, cell_seed=0)
    assert ok
    assert any("repetition_penalty=1.05" in m for m in messages)


def test_sample_seed_for_draw_distinct():
    seeds = [sample_seed_for_draw(0, i) for i in range(5)]
    assert len(set(seeds)) == 5
