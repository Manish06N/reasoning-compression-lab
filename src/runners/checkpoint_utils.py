"""Atomic JSONL checkpoints and archive backup helpers."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write JSONL atomically so a power cut mid-write cannot corrupt the main file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_jsonl(path: Path) -> tuple[bool, int]:
    """Return (valid, row_count). Invalid if any line fails JSON parse."""
    if not path.exists():
        return True, 0
    count = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                json.loads(line)
                count += 1
        return True, count
    except json.JSONDecodeError:
        return False, count


def recover_jsonl_from_backup(path: Path, backup_dir: Path) -> bool:
    """If path is corrupt/missing, restore from backup latest mirror."""
    latest = backup_dir / "latest" / "raw" / path.name
    if not latest.exists():
        return False
    ok, _ = validate_jsonl(latest)
    if not ok:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, path)
    return True


def write_progress(
    output_root: Path,
    cell_id: str,
    rows_done: int,
    rows_total: int,
    status: str = "in_progress",
) -> None:
    progress_dir = output_root / "checkpoints"
    progress_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cell_id": cell_id,
        "rows_done": rows_done,
        "rows_total": rows_total,
        "status": status,
        "updated_at": utc_now_iso(),
    }
    progress_path = progress_dir / f"{cell_id}.json"
    tmp = progress_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(progress_path)


def backup_file(src: Path, backup_root: Path, subdir: str) -> None:
    """Mirror one artifact under backup_root/latest/{subdir}/."""
    if not src.exists():
        return
    dest_dir = backup_root / "latest" / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / src.name)


def backup_mirror(backup_root: Path, output_root: Path) -> None:
    """Sync archive into backup_root/latest/ only (no timestamped snapshot)."""
    latest = backup_root / "latest"
    for sub in ("raw", "scored", "results", "logs", "checkpoints"):
        src_dir = output_root / sub
        if not src_dir.exists():
            continue
        dest = latest / sub
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src_dir, dest)
    for name in ("manifest.json", "state.json"):
        src = output_root / name
        if src.exists():
            latest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, latest / name)


def backup_snapshot(backup_root: Path, output_root: Path, label: str | None = None) -> Path:
    """Mirror to latest/ and copy to snapshots/{timestamp}/."""
    backup_mirror(backup_root, output_root)
    stamp = label or datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot = backup_root / "snapshots" / stamp
    latest = backup_root / "latest"
    if latest.exists():
        if snapshot.exists():
            shutil.rmtree(snapshot)
        shutil.copytree(latest, snapshot)
    return snapshot


def update_state(output_root: Path, **fields: Any) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "state.json"
    lock_path = output_root / "state.json.lock"
    state: dict[str, Any] = {}
    with lock_path.open("a", encoding="utf-8") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
            state.update(fields)
            state["updated_at"] = utc_now_iso()

            fd, tmp_name = tempfile.mkstemp(
                prefix=f"{state_path.name}.",
                suffix=".tmp",
                dir=output_root,
                text=True,
            )
            tmp = Path(tmp_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(json.dumps(state, indent=2))
                    f.flush()
                    os.fsync(f.fileno())
                tmp.replace(state_path)
            finally:
                if tmp.exists():
                    tmp.unlink()
        finally:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
