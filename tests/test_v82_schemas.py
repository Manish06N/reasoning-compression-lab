"""Tests for prompt profiles and provenance."""

from src.runners.config_utils import load_cell_config, resolve_prompt_template
from src.schemas.provenance import config_hash, provenance_fields


def test_level_a_uses_qrm_reproduction_prompt():
    cell = load_cell_config("configs/cells/level_a_bf16_seed0.json")
    assert cell["prompt_profile"] == "reproduction"
    assert cell["task"]["prompt_template_file"] == "prompts/qrm_math500.txt"


def test_sober_profile_math500():
    task = {"task_name": "math500", "prompt_template_file": "prompts/math500.txt"}
    cell = {"prompt_profile": "sober"}
    assert resolve_prompt_template(task, cell) == "prompts/math500.txt"


def test_provenance_fields_present():
    cell = load_cell_config("configs/cells/level_a_bf16_seed0.json")
    fields = provenance_fields(cell, prompt_template_file=cell["task"]["prompt_template_file"])
    assert fields["git_commit"]
    assert fields["config_hash"] == config_hash(cell)
    assert fields["schema_version"] == "raw_response.v1"
