"""Multisample inference rows must include full provenance."""

from src.runners.config_utils import load_cell_config
from src.runners.raw_row import build_raw_response_row
from src.schemas.validate import validate_row


def test_multisample_row_has_provenance_fields():
    cell = load_cell_config("configs/cells/level_a_bf16_seed0.json")
    row = build_raw_response_row(
        row_base={"id": "1", "problem": "x", "gold_solution": "1"},
        result={
            "prompt": "p",
            "completion": "c",
            "latency_sec": 0.5,
            "peak_vram_gb": 8.0,
            "prompt_tokens": 1,
            "completion_tokens": 2,
        },
        cell=cell,
        prompt_template_file=cell["task"]["prompt_template_file"],
        sample_index=0,
        sample_seed=42,
        n_samples=5,
    )
    for field in (
        "run_id",
        "git_commit",
        "config_hash",
        "prompt_template_version",
        "schema_version",
        "sample_index",
        "n_samples",
    ):
        assert field in row, f"missing {field}"
    assert validate_row(row) == []
