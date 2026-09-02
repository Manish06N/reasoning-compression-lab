"""Tests for resume guard (bad archive / stale decoding / RunSpec hash)."""

import json
import os
from pathlib import Path
from unittest import mock

from src.runners.config_utils import load_cell_config
from src.runners.resume_guard import (
    INVALID_FOR_PUBLICATION_MARKER,
    archive_is_forbidden,
    resume_block_reason,
)
from src.runners.run_spec import run_spec_from_cell
from src.schemas.provenance import provenance_fields


def test_forbidden_archive_marker():
    p = Path("/scratch/user/reasoning-compression-lab/outputs-hpc-2a100-main-2026-06-29/raw/x.jsonl")
    assert archive_is_forbidden(p) is True
    p2 = Path("/scratch/user/reasoning-compression-lab/outputs-hpc-2a100-main-2026-07-01-rerun/raw/x.jsonl")
    assert archive_is_forbidden(p2) is False


def test_forbidden_archive_marker_file(tmp_path):
    archive = tmp_path / "outputs-clean"
    archive.mkdir()
    raw = archive / "raw" / "cell.jsonl"
    raw.parent.mkdir()
    raw.write_text("{}\n", encoding="utf-8")
    assert archive_is_forbidden(raw) is False

    (archive / INVALID_FOR_PUBLICATION_MARKER).write_text("pre-fix decoding\n", encoding="utf-8")
    assert archive_is_forbidden(raw) is True


def test_forbidden_archive_env_patterns(tmp_path):
    archive = tmp_path / "outputs-custom-bad-run"
    archive.mkdir()
    path = archive / "raw" / "cell.jsonl"
    path.parent.mkdir()
    path.write_text("{}\n", encoding="utf-8")
    with mock.patch.dict(os.environ, {"QREASON_FORBIDDEN_ARCHIVE_PATTERNS": "custom-bad-run"}):
        assert archive_is_forbidden(path) is True
    assert archive_is_forbidden(path) is False


def test_blocks_resume_without_repetition_penalty(tmp_path):
    out = tmp_path / "level_a.jsonl"
    row = {
        "cell_id": "level_a_qwen7b_bf16_math500_seed0",
        "decoding_repetition_penalty": None,
        "git_commit": "abc123",
        "config_hash": "deadbeef",
    }
    out.write_text(json.dumps(row) + "\n", encoding="utf-8")
    cell = {
        "cell_id": "level_a_qwen7b_bf16_math500_seed0",
        "model_config": "configs/legacy_models/deepseek_r1_qwen_7b.json",
        "task_config": "configs/tasks/math500.json",
        "quant_config": "bf16",
        "seed": 0,
        "decoding": {"repetition_penalty": 1.05, "temperature": 0.6},
        "model_path": "/tmp/model",
        "prompt_profile": "reproduction",
    }
    reason = resume_block_reason(out, cell, allow_resume=False)
    assert reason is not None
    assert "repetition_penalty" in reason


def test_allow_resume_bypasses_block(tmp_path):
    out = tmp_path / "level_a.jsonl"
    out.write_text(json.dumps({"decoding_repetition_penalty": None}) + "\n")
    cell = {"decoding": {"repetition_penalty": 1.05}}
    assert resume_block_reason(out, cell, allow_resume=True) is None


def test_maj5_resume_accepts_matching_n_samples(tmp_path):
    cell = load_cell_config("configs/cells/level_a_bf16_seed0.json")
    prompt_file = cell["task"]["prompt_template_file"]
    run_spec = run_spec_from_cell(
        cell,
        prompt_template_file=prompt_file,
        batch_size=1,
        n_samples=5,
        max_model_len=cell["model"].get("max_model_len"),
    )
    prov = provenance_fields(cell, run_spec=run_spec)
    out = tmp_path / "level_a_maj5.jsonl"
    row = {
        **prov,
        "id": "x",
        "sample_index": 0,
        "git_commit": prov["git_commit"],
        "decoding_repetition_penalty": cell["decoding"].get("repetition_penalty"),
    }
    out.write_text(json.dumps(row) + "\n")
    assert resume_block_reason(out, cell, allow_resume=False, run_spec=run_spec) is None


def test_maj5_resume_rejects_wrong_n_samples(tmp_path):
    cell = load_cell_config("configs/cells/level_a_bf16_seed0.json")
    prompt_file = cell["task"]["prompt_template_file"]
    archive_spec = run_spec_from_cell(
        cell,
        prompt_template_file=prompt_file,
        batch_size=1,
        n_samples=5,
        max_model_len=cell["model"].get("max_model_len"),
    )
    resume_spec = run_spec_from_cell(
        cell,
        prompt_template_file=prompt_file,
        batch_size=1,
        n_samples=None,
        max_model_len=cell["model"].get("max_model_len"),
    )
    prov = provenance_fields(cell, run_spec=archive_spec)
    out = tmp_path / "level_a_maj5.jsonl"
    row = {
        **prov,
        "id": "x",
        "sample_index": 0,
        "git_commit": prov["git_commit"],
        "decoding_repetition_penalty": cell["decoding"].get("repetition_penalty"),
    }
    out.write_text(json.dumps(row) + "\n")
    reason = resume_block_reason(out, cell, allow_resume=False, run_spec=resume_spec)
    assert reason is not None
    assert "config_hash mismatch" in reason


def test_resume_allowed_when_only_output_commits_moved_head(monkeypatch, tmp_path):
    from src.runners.config_utils import load_cell_config
    from src.runners.run_spec import run_spec_from_cell
    from src.schemas.provenance import provenance_fields

    cell = load_cell_config("configs/cells/level_a_bf16_seed0.json")
    prompt_file = cell["task"]["prompt_template_file"]
    run_spec = run_spec_from_cell(
        cell,
        prompt_template_file=prompt_file,
        batch_size=1,
        n_samples=None,
        max_model_len=cell["model"].get("max_model_len"),
    )
    prov = provenance_fields(cell, run_spec=run_spec)
    out = tmp_path / "level_a.jsonl"
    row = {
        **prov,
        "id": "x",
        "git_commit": "oldcommit",
        "decoding_repetition_penalty": cell["decoding"].get("repetition_penalty"),
    }
    out.write_text(json.dumps(row) + "\n")

    monkeypatch.setattr(
        "src.runners.resume_guard.code_changed_since",
        lambda _root, commit: commit != "oldcommit",
    )
    monkeypatch.setattr("src.runners.resume_guard.git_commit_short", lambda: "newcommit")

    reason = resume_block_reason(out, cell, allow_resume=False, run_spec=run_spec)
    assert reason is None


def test_resume_blocked_when_code_changed(monkeypatch, tmp_path):
    from src.runners.config_utils import load_cell_config
    from src.runners.run_spec import run_spec_from_cell
    from src.schemas.provenance import provenance_fields

    cell = load_cell_config("configs/cells/level_a_bf16_seed0.json")
    prompt_file = cell["task"]["prompt_template_file"]
    run_spec = run_spec_from_cell(
        cell,
        prompt_template_file=prompt_file,
        batch_size=1,
        n_samples=None,
        max_model_len=cell["model"].get("max_model_len"),
    )
    prov = provenance_fields(cell, run_spec=run_spec)
    out = tmp_path / "level_a.jsonl"
    row = {
        **prov,
        "id": "x",
        "git_commit": "oldcommit",
        "decoding_repetition_penalty": cell["decoding"].get("repetition_penalty"),
    }
    out.write_text(json.dumps(row) + "\n")

    monkeypatch.setattr("src.runners.resume_guard.code_changed_since", lambda _root, _commit: True)
    monkeypatch.setattr("src.runners.resume_guard.git_commit_short", lambda: "newcommit")

    reason = resume_block_reason(out, cell, allow_resume=False, run_spec=run_spec)
    assert reason is not None
    assert "code changed since row git" in reason
