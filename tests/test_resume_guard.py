"""Tests for resume guard (bad archive / stale decoding)."""

import json
from pathlib import Path

from src.runners.resume_guard import (
    archive_is_forbidden,
    resume_block_reason,
)


def test_forbidden_archive_marker():
    p = Path("/scratch/user/reasoning-compression-lab/outputs-hpc-2a100-main-2026-06-29/raw/x.jsonl")
    assert archive_is_forbidden(p) is True
    p2 = Path("/scratch/user/reasoning-compression-lab/outputs-hpc-2a100-main-2026-07-01-rerun/raw/x.jsonl")
    assert archive_is_forbidden(p2) is False


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
        "model_config": "configs/models/deepseek_r1_qwen_7b.json",
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
