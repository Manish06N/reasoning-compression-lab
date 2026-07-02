"""Schema validation for raw inference rows."""

from src.runners.config_utils import load_cell_config
from src.runners.raw_row import build_raw_response_row
from src.schemas.validate import validate_row


def _minimal_result() -> dict:
    return {
        "prompt": "test prompt",
        "completion": "test completion",
        "latency_sec": 1.0,
        "peak_vram_gb": 10.0,
        "prompt_tokens": 5,
        "completion_tokens": 10,
        "finish_reason": "stop",
        "truncated": False,
    }


def test_build_raw_response_row_passes_schema():
    cell = load_cell_config("configs/cells/level_a_bf16_seed0.json")
    row_base = {
        "id": "0",
        "problem": "2+2?",
        "gold_solution": "4",
    }
    row = build_raw_response_row(
        row_base=row_base,
        result=_minimal_result(),
        cell=cell,
        prompt_template_file=cell["task"]["prompt_template_file"],
        validate=True,
    )
    errors = validate_row(row)
    assert errors == [], errors


def test_validate_row_rejects_missing_required():
    errors = validate_row({"id": "0"})
    assert errors
