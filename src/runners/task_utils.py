"""Task row counts and completion helpers shared by orchestration scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.runners.config_utils import REPO_ROOT, load_json

# HuggingFace split sizes for publication cells (seed 0, full split).
TASK_ROW_COUNTS: dict[str, int] = {
    "math500": 500,
    "gsm8k": 1319,
    "gpqa_diamond": 198,
}


def expected_rows_for_task(task_name: str, limit: Optional[int] = None) -> int:
    if limit is not None:
        return limit
    if task_name in TASK_ROW_COUNTS:
        return TASK_ROW_COUNTS[task_name]
    if task_name.startswith("gpqa"):
        return TASK_ROW_COUNTS["gpqa_diamond"]
    return 1


def expected_rows_for_cell(cell_config_path: str | Path, limit: Optional[int] = None) -> int:
    path = Path(cell_config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    cell = load_json(path)
    task = load_json(cell["task_config"])
    return expected_rows_for_task(task["task_name"], limit=limit)
