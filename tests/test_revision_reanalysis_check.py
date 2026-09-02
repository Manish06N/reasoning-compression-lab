"""Fast checks for the canonical reanalysis comparator (no 10k bootstrap)."""

from scripts.analysis.revision_reanalysis import json_diff


def test_json_diff_identical():
    assert json_diff({"a": 1.0, "b": True}, {"a": 1.0, "b": True}) == []


def test_json_diff_float_tolerance():
    assert json_diff({"x": 1.0}, {"x": 1.0 + 1e-12}) == []
    diffs = json_diff({"x": 1.0}, {"x": 1.001})
    assert diffs and "$.x" in diffs[0]


def test_json_diff_tost_flag():
    diffs = json_diff({"tost_equiv_1pp": False}, {"tost_equiv_1pp": True})
    assert len(diffs) == 1
    assert "tost_equiv_1pp" in diffs[0]


def test_json_diff_missing_key():
    diffs = json_diff({"loops": 25}, {})
    assert any("missing" in d for d in diffs)


def test_canonical_pass1_from_frozen_report():
    from scripts.analysis.revision_reanalysis import load_canonical_pass1

    pass1 = load_canonical_pass1()
    assert pass1[("Qwen-7B", "BF16")] == 0.94
    assert pass1[("Qwen-7B", "FP8")] == 0.944
    assert pass1[("Qwen-7B", "AWQ-4")] == 0.9312
    assert pass1[("Llama-8B", "AWQ-4")] == 0.8648
    assert len(pass1) == 8
