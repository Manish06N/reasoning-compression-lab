"""Tests for QRM serving parity helpers."""

from src.runners.vllm_serving import build_llm_init_kwargs, load_serving_defaults


def test_build_llm_init_kwargs_qrm_defaults():
    model_cfg = {
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "tensor_parallel_size": 1,
        "trust_remote_code": True,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.9,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
    }
    kwargs = build_llm_init_kwargs("/tmp/model", model_cfg, seed=42)
    assert kwargs["seed"] == 42
    assert kwargs["gpu_memory_utilization"] == 0.9
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["enable_chunked_prefill"] is False
    assert kwargs["enforce_eager"] is True
    assert kwargs["max_model_len"] == 32768


def test_load_serving_defaults_has_qrm_keys():
    defaults = load_serving_defaults()
    assert defaults["gpu_memory_utilization"] == 0.9
    assert defaults["enable_prefix_caching"] is False
    assert defaults["enable_chunked_prefill"] is False