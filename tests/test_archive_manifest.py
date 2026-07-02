"""Tests for locked archive manifest updates."""

from __future__ import annotations

import threading
from pathlib import Path

from src.runners.archive_manifest import upsert_manifest_cell, upsert_manifest_header


def test_concurrent_manifest_cell_upserts(tmp_path: Path):
    archive = tmp_path / "archive"
    archive.mkdir()

    def writer(cell_id: str) -> None:
        upsert_manifest_cell(
            archive,
            {
                "cell_id": cell_id,
                "status": "in_progress",
                "raw": str(archive / "raw" / f"{cell_id}.jsonl"),
                "summary": None,
                "metadata": str(archive / "metadata" / f"{cell_id}.json"),
                "rows_saved": 1,
                "updated_at": f"2026-07-02T00:00:00+00:00-{cell_id}",
            },
        )

    upsert_manifest_header(archive, {"block_id": "b01", "updated_at": "2026-07-02T00:00:00+00:00"})
    threads = [threading.Thread(target=writer, args=(f"cell_{idx}",)) for idx in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    manifest = (archive / "manifest.json").read_text(encoding="utf-8")
    for idx in range(4):
        assert f"cell_{idx}" in manifest
