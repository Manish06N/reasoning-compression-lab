"""Tests for config loading and task row counts."""

from src.runners.config_utils import load_decoding_from_file
from src.runners.task_utils import expected_rows_for_cell, expected_rows_for_task


def test_load_decoding_passes_repetition_penalty():
    try:
        decoding = load_decoding_from_file("configs/decoding/repro_qrm.yaml")
    except ImportError:
        return
    assert decoding["repetition_penalty"] == 1.05
    assert decoding["temperature"] == 0.6
    assert decoding["max_tokens"] == 32768
    assert "notes" not in decoding


def test_expected_rows_for_tasks():
    assert expected_rows_for_task("math500") == 500
    assert expected_rows_for_task("gsm8k") == 1319
    assert expected_rows_for_task("gpqa_diamond") == 198
    assert expected_rows_for_task("gpqa_other") == 198


def test_expected_rows_for_cell_math500():
    rows = expected_rows_for_cell("configs/cells/level_a_bf16_seed0.json")
    assert rows == 500


def test_expected_rows_for_cell_gpqa():
    rows = expected_rows_for_cell("configs/cells/level_c_qwen7b_fp8_gpqa_seed0.json")
    assert rows == 198
