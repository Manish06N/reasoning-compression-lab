#!/usr/bin/env python3
"""Python checks for stale raw JSONL in a publication archive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.runners.checkpoint_utils import load_jsonl
from src.runners.config_utils import load_cell_config
from src.runners.resume_guard import allow_resume_from_env, archive_is_forbidden, resume_block_reason


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()

    archive = Path(args.archive)
    if not archive.is_absolute():
        archive = ROOT / archive

    if archive_is_forbidden(archive) and not allow_resume_from_env():
        raise SystemExit(
            f"ERROR: forbidden archive path {archive}. "
            "Delete it or choose a new QREASON_OUTPUT_ROOT."
        )

    raw_dir = archive / "raw"
    if not raw_dir.exists():
        print(f"No raw/ yet under {archive} — ok for fresh run.")
        return

    allow = allow_resume_from_env()
    problems: list[str] = []
    for jsonl in sorted(raw_dir.glob("*.jsonl")):
        cell_id = jsonl.stem
        cell_cfg = None
        for path in (ROOT / "configs/cells").glob("*.json"):
            cfg = load_cell_config(path.relative_to(ROOT).as_posix())
            if cfg.get("cell_id") == cell_id:
                cell_cfg = cfg
                break
        if cell_cfg is None:
            rows = load_jsonl(jsonl)
            if rows and not allow:
                problems.append(f"{jsonl.name}: unknown cell, {len(rows)} rows (cannot validate)")
            continue
        reason = resume_block_reason(jsonl, cell_cfg, allow_resume=allow)
        if reason:
            problems.append(reason)

    if problems:
        msg = "ERROR: unsafe resume detected:\n" + "\n".join(f"  - {p}" for p in problems)
        msg += "\nFix: rm raw JSONL or entire archive, or export QREASON_ALLOW_RESUME=1"
        raise SystemExit(msg)

    print(f"Checked {len(list(raw_dir.glob('*.jsonl')))} raw file(s) — ok to resume or start.")


if __name__ == "__main__":
    main()
