"""Locked manifest and per-cell metadata updates for HPC output archives."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.runners.checkpoint_utils import atomic_locked_json_update


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(os.environ.get("QR", Path(__file__).resolve().parents[2]))


def _git_cmd(args: list[str], repo: Path) -> str | None:
    try:
        return subprocess.check_output(args, cwd=repo, text=True).strip()
    except Exception:
        return None


def upsert_manifest_header(
    archive_root: Path,
    header_fields: dict[str, Any],
    *,
    repo: Path | None = None,
) -> dict[str, Any]:
    """Merge block/job header fields into manifest.json under a file lock."""
    repo = repo or _repo_root()
    manifest_path = archive_root / "manifest.json"
    now = header_fields.get("updated_at") or utc_now_iso()
    header_fields = {**header_fields, "updated_at": now}

    def mutator(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("cells") is not None and state.get("started_at"):
            merged = {**state, **header_fields, "resumed_at": now}
            merged.setdefault("cells", state.get("cells", []))
            return merged
        default = {
            "machine": "PARAM Rudra HPC 2x A100",
            "output_root": str(archive_root),
            "publication_mode": True,
            "protocol": "hpc_repro_qrm",
            "started_at": now,
            "cells": [],
        }
        return {**default, **header_fields}

    return atomic_locked_json_update(manifest_path, mutator, default={"cells": []})


def upsert_manifest_cell(archive_root: Path, cell_entry: dict[str, Any]) -> dict[str, Any]:
    """Upsert one cell entry in manifest.json under a file lock."""
    manifest_path = archive_root / "manifest.json"
    cell_id = cell_entry["cell_id"]

    def mutator(state: dict[str, Any]) -> dict[str, Any]:
        cells = [c for c in state.get("cells", []) if c.get("cell_id") != cell_id]
        cells.append(cell_entry)
        state["cells"] = cells
        state["updated_at"] = cell_entry.get("updated_at", utc_now_iso())
        return state

    return atomic_locked_json_update(manifest_path, mutator, default={"cells": []})


def write_cell_metadata_file(archive_root: Path, cell_id: str, payload: dict[str, Any]) -> Path:
    metadata_dir = archive_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"{cell_id}.json"
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metadata_path


def build_header_fields(
    *,
    block_id: str,
    block_file: str | None,
    decoding_config: str,
    batch_size: int,
    checkpoint_every: int,
    date_tag: str | None,
    repo: Path | None = None,
) -> dict[str, Any]:
    repo = repo or _repo_root()
    now = utc_now_iso()
    return {
        "date": date_tag,
        "block_id": block_id,
        "block_file": block_file,
        "decoding_config": decoding_config,
        "batch_size": batch_size,
        "checkpoint_every": checkpoint_every,
        "git_commit": _git_cmd(["git", "rev-parse", "HEAD"], repo),
        "git_branch": _git_cmd(["git", "branch", "--show-current"], repo),
        "git_status_short": _git_cmd(["git", "status", "--short"], repo),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_node_list": os.environ.get("SLURM_NODELIST"),
        "user": os.environ.get("USER"),
        "hostname": platform.node(),
        "updated_at": now,
    }


def build_cell_payload(
    *,
    cell_id: str,
    cell_cfg_path: Path,
    gpu_id: str,
    status: str,
    raw_path: Path,
    summary_path: Path | None,
    decoding_config: str,
    batch_size: int,
    checkpoint_every: int,
    repo: Path | None = None,
) -> dict[str, Any]:
    repo = repo or _repo_root()
    cell = json.loads((repo / cell_cfg_path).read_text(encoding="utf-8"))
    model = json.loads((repo / cell["model_config"]).read_text(encoding="utf-8"))
    task = json.loads((repo / cell["task_config"]).read_text(encoding="utf-8"))
    decoding_abs = repo / decoding_config
    decoding_text = decoding_abs.read_text(encoding="utf-8") if decoding_abs.exists() else ""
    rows = 0
    if raw_path.exists():
        rows = sum(1 for line in raw_path.open("r", encoding="utf-8") if line.strip())
    now = utc_now_iso()
    return {
        "cell_id": cell_id,
        "status": status,
        "updated_at": now,
        "gpu_id": gpu_id,
        "rows_saved": rows,
        "raw": str(raw_path),
        "summary": str(summary_path) if summary_path else None,
        "cell_config_path": str(cell_cfg_path),
        "cell_config": cell,
        "model_config_path": cell["model_config"],
        "model_config": model,
        "task_config_path": cell["task_config"],
        "task_config": task,
        "decoding_config_path": decoding_config,
        "decoding_config_text": decoding_text,
        "batch_size": batch_size,
        "checkpoint_every": checkpoint_every,
        "git_commit": _git_cmd(["git", "rev-parse", "HEAD"], repo),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_node_list": os.environ.get("SLURM_NODELIST"),
    }


def update_cell_metadata(
    archive_root: Path,
    *,
    cell_id: str,
    cell_cfg_path: Path,
    gpu_id: str,
    status: str,
    raw_path: Path,
    summary_path: Path | None,
    decoding_config: str,
    batch_size: int,
    checkpoint_every: int,
    repo: Path | None = None,
) -> Path:
    payload = build_cell_payload(
        cell_id=cell_id,
        cell_cfg_path=cell_cfg_path,
        gpu_id=gpu_id,
        status=status,
        raw_path=raw_path,
        summary_path=summary_path,
        decoding_config=decoding_config,
        batch_size=batch_size,
        checkpoint_every=checkpoint_every,
        repo=repo,
    )
    metadata_path = write_cell_metadata_file(archive_root, cell_id, payload)
    upsert_manifest_cell(
        archive_root,
        {
            "cell_id": cell_id,
            "status": payload["status"],
            "raw": payload["raw"],
            "summary": payload["summary"],
            "metadata": str(metadata_path),
            "rows_saved": payload["rows_saved"],
            "updated_at": payload["updated_at"],
        },
    )
    return metadata_path


def _cmd_header(args: argparse.Namespace) -> None:
    archive = Path(args.archive)
    repo = Path(args.repo) if args.repo else _repo_root()
    fields = build_header_fields(
        block_id=args.block_id,
        block_file=args.block_file,
        decoding_config=args.decoding,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        date_tag=args.date_tag,
        repo=repo,
    )
    upsert_manifest_header(archive, fields, repo=repo)


def _cmd_cell_metadata(args: argparse.Namespace) -> None:
    archive = Path(args.archive)
    repo = Path(args.repo) if args.repo else _repo_root()
    summary = Path(args.summary) if args.summary else None
    update_cell_metadata(
        archive,
        cell_id=args.cell_id,
        cell_cfg_path=Path(args.cell_config),
        gpu_id=args.gpu_id,
        status=args.status,
        raw_path=Path(args.raw),
        summary_path=summary,
        decoding_config=args.decoding,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        repo=repo,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    header = sub.add_parser("header")
    header.add_argument("--archive", required=True)
    header.add_argument("--block-id", required=True)
    header.add_argument("--block-file", default=None)
    header.add_argument("--decoding", default="configs/decoding/repro_qrm.yaml")
    header.add_argument("--batch-size", type=int, default=1)
    header.add_argument("--checkpoint-every", type=int, default=10)
    header.add_argument("--date-tag", default=None)
    header.add_argument("--repo", default=None)
    header.set_defaults(func=_cmd_header)

    cell = sub.add_parser("cell-metadata")
    cell.add_argument("--archive", required=True)
    cell.add_argument("--cell-id", required=True)
    cell.add_argument("--cell-config", required=True)
    cell.add_argument("--gpu-id", default="0")
    cell.add_argument("--status", required=True)
    cell.add_argument("--raw", required=True)
    cell.add_argument("--summary", default="")
    cell.add_argument("--decoding", default="configs/decoding/repro_qrm.yaml")
    cell.add_argument("--batch-size", type=int, default=1)
    cell.add_argument("--checkpoint-every", type=int, default=10)
    cell.add_argument("--repo", default=None)
    cell.set_defaults(func=_cmd_cell_metadata)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
