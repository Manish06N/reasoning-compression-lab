"""Tests for publication mode helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def test_assert_clean_git_tree_missing_git(monkeypatch, tmp_path: Path):
    from src.runners.publication_mode import assert_clean_git_tree

    def fake_run(*_args, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="requires Git installed"):
        assert_clean_git_tree(tmp_path)


def test_assert_code_paths_clean_ignores_output_changes(monkeypatch, tmp_path: Path):
    from src.runners.publication_mode import assert_code_paths_clean

    calls: list[list[str]] = []

    def fake_run(args, **_kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert_code_paths_clean(tmp_path)
    assert all("src" in call for call in calls)


def test_assert_code_paths_clean_blocks_dirty_code(monkeypatch, tmp_path: Path):
    from src.runners.publication_mode import assert_code_paths_clean

    def fake_run(args, **_kwargs):
        if args[:3] == ["git", "diff", "--quiet"] and "src" in args:
            raise subprocess.CalledProcessError(1, args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="uncommitted changes in code paths"):
        assert_code_paths_clean(tmp_path)


def test_warn_or_assert_code_paths_clean_warns_by_default(monkeypatch, tmp_path: Path, capsys):
    from src.runners.publication_mode import warn_or_assert_code_paths_clean

    def fake_run(args, **_kwargs):
        if args[:3] == ["git", "diff", "--quiet"] and "src" in args:
            raise subprocess.CalledProcessError(1, args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.delenv("QREASON_STRICT_GIT", raising=False)

    warn_or_assert_code_paths_clean(tmp_path)
    captured = capsys.readouterr()
    assert "WARN:" in captured.err
    assert "QREASON_STRICT_GIT=1" in captured.err


def test_warn_or_assert_code_paths_clean_strict_exits(monkeypatch, tmp_path: Path):
    from src.runners.publication_mode import warn_or_assert_code_paths_clean

    def fake_run(args, **_kwargs):
        if args[:3] == ["git", "diff", "--quiet"] and "src" in args:
            raise subprocess.CalledProcessError(1, args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("QREASON_STRICT_GIT", "1")

    with pytest.raises(SystemExit, match="uncommitted changes in code paths"):
        warn_or_assert_code_paths_clean(tmp_path)
