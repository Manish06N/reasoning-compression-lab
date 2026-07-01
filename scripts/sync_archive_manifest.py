#!/usr/bin/env python3
"""Refresh manifest.json, metadata/*.json, and state.json from archive files on disk."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runners.task_utils import expected_rows_for_task


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.open("r", encoding="utf-8") if line.strip())


def _task_name_from_meta(meta: dict) -> str:
    task_cfg = meta.get("task_config") or {}
    if isinstance(task_cfg, dict) and task_cfg.get("task_name"):
        return str(task_cfg["task_name"])
    cell_cfg = meta.get("cell_config") or {}
    task_path = cell_cfg.get("task_config")
    if task_path:
        task_file = Path(task_path)
        if not task_file.is_absolute():
            task_file = ROOT / task_file
        if task_file.exists():
            return json.loads(task_file.read_text(encoding="utf-8")).get("task_name", "math500")
    return "math500"


def sync_archive(archive: Path) -> None:
    manifest_path = archive / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"cells": []}
    cells_out = []
    last_cell_id = None
    last_rows_done = 0
    last_rows_total = 500
    last_phase = "idle"

    for meta_path in sorted((archive / "metadata").glob("*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cell_id = meta["cell_id"]
        raw_name = f"{cell_id}.jsonl"
        raw_path = archive / "raw" / raw_name
        rows = count_jsonl_rows(raw_path)
        summary_path = archive / "results" / f"{cell_id}_summary.json"
        scored_path = archive / "scored" / raw_name
        want = expected_rows_for_task(_task_name_from_meta(meta))

        if rows >= want and summary_path.exists() and scored_path.exists():
            status = "scored"
            last_phase = "scored"
        elif rows >= want and summary_path.exists():
            status = "completed"
            last_phase = "completed"
        elif rows > 0:
            status = "in_progress"
            last_phase = "inference"
        else:
            status = meta.get("status", "pending")

        meta["rows_saved"] = rows
        meta["status"] = status
        meta["updated_at"] = utc_now_iso()
        meta["raw"] = str(raw_path)
        meta["summary"] = str(summary_path) if summary_path.exists() else None
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        cells_out.append(
            {
                "cell_id": cell_id,
                "status": status,
                "raw": str(raw_path),
                "summary": str(summary_path) if summary_path.exists() else None,
                "metadata": str(meta_path),
                "rows_saved": rows,
                "updated_at": meta["updated_at"],
            }
        )
        if rows >= last_rows_done:
            last_cell_id = cell_id
            last_rows_done = rows
            last_rows_total = want

    manifest["cells"] = cells_out
    manifest["updated_at"] = utc_now_iso()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    state = {
        "last_cell_id": last_cell_id,
        "last_phase": last_phase,
        "rows_done": last_rows_done,
        "rows_total": last_rows_total,
        "updated_at": utc_now_iso(),
    }
    (archive / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"Synced manifest ({len(cells_out)} cells) and state.json for {archive}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()
    archive = Path(args.archive)
    if not archive.is_absolute():
        archive = ROOT / archive
    sync_archive(archive)


if __name__ == "__main__":
    main()
