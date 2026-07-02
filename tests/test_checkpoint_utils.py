"""Tests for checkpoint backup helpers."""

from __future__ import annotations

from pathlib import Path

from src.runners.checkpoint_utils import backup_mirror


def test_backup_mirror_ignores_tmp_files(tmp_path: Path):
    output = tmp_path / "archive"
    raw = output / "raw"
    raw.mkdir(parents=True)
    (raw / "cell.jsonl").write_text("{}\n", encoding="utf-8")
    tmp_file = raw / "cell.jsonl.tmp"
    tmp_file.write_text("partial\n", encoding="utf-8")
    tmp_file.unlink()

    backup = tmp_path / "backup"
    backup_mirror(backup, output)

    assert (backup / "latest" / "raw" / "cell.jsonl").exists()
