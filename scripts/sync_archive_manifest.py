#!/usr/bin/env python3
"""Refresh an archive manifest from raw/scored/results files on disk."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync archive manifest/state from disk.")
    parser.add_argument("archive", help="Archive root containing raw/, scored/, and results/.")
    args = parser.parse_args()

    archive = Path(args.archive)
    if not archive.is_absolute():
        archive = ROOT / archive
    manifest_path = archive / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cells = manifest.setdefault("cells", {})

    for raw in sorted((archive / "raw").glob("*.jsonl")):
        cell_id = raw.stem
        scored = archive / "scored" / raw.name
        summary = archive / "results" / f"{cell_id}_summary.json"
        raw_rows = _count_jsonl(raw)
        scored_rows = _count_jsonl(scored)
        status = "scored" if scored_rows >= raw_rows and summary.exists() else "inference_completed"
        cells[cell_id] = {**cells.get(cell_id, {}), "status": status, "raw": _rel(raw), "scored": _rel(scored) if scored.exists() else "", "summary": _rel(summary) if summary.exists() else "", "rows": raw_rows, "scored_rows": scored_rows}

    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    state_path = archive / "state.json"
    state_path.write_text(json.dumps({"updated_at": manifest["updated_at"], "num_cells": len(cells)}, indent=2), encoding="utf-8")
    print(f"Synced {len(cells)} cells in {manifest_path}")


if __name__ == "__main__":
    main()
