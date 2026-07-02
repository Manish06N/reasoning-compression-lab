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


def test_sober_profile_gsm8k():
    task = {"task_name": "gsm8k", "prompt_template_file": "prompts/gsm8k.txt"}
    cell = {"prompt_profile": "sober"}
    assert resolve_prompt_template(task, cell) == "prompts/gsm8k.txt"


def test_level_b_gsm8k_resolves_sober_prompt():
    cell = load_cell_config("configs/cells/level_b_qwen7b_fp8_gsm8k_seed0.json")
    assert cell["prompt_profile"] == "sober"
    assert cell["task"]["prompt_template_file"] == "prompts/gsm8k.txt"


def test_provenance_fields_present():
    cell = load_cell_config("configs/cells/level_a_bf16_seed0.json")
    fields = provenance_fields(cell, prompt_template_file=cell["task"]["prompt_template_file"])
    assert fields["git_commit"]
    assert fields["config_hash"] == config_hash(
        cell, prompt_template_file=cell["task"]["prompt_template_file"]
    )
    assert fields["schema_version"] == "raw_response.v1"
    assert fields.get("dataset_revision") == "main"


def test_config_hash_independent_of_model_path():
    cell = load_cell_config("configs/cells/level_a_bf16_seed0.json")
    tmpl = cell["task"]["prompt_template_file"]
    h1 = config_hash(cell, prompt_template_file=tmpl)
    alt = dict(cell)
    alt["model_path"] = "/totally/different/path"
    h2 = config_hash(alt, prompt_template_file=tmpl)
    assert h1 == h2


def test_config_hash_changes_with_batch_size():
    cell = load_cell_config("configs/cells/level_a_bf16_seed0.json")
    tmpl = cell["task"]["prompt_template_file"]
    h1 = config_hash(cell, prompt_template_file=tmpl, batch_size=1)
    h2 = config_hash(cell, prompt_template_file=tmpl, batch_size=4)
    assert h1 != h2
