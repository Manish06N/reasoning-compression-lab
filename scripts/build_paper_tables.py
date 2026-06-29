#!/usr/bin/env python3
"""Build publication CSV tables from an experiment output archive."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

MAIN_FIELDS = [
    "cell_id",
    "task",
    "model_family",
    "model_id",
    "quant_config",
    "seed",
    "n",
    "pass_at_1",
    "pass_at_1_ci95_low",
    "pass_at_1_ci95_high",
    "num_correct",
]

EFFICIENCY_FIELDS = [
    "cell_id",
    "task",
    "model_family",
    "quant_config",
    "latency_sec_mean",
    "latency_sec_p50",
    "peak_vram_gb_max",
    "peak_vram_gb_mean",
    "prompt_tokens_mean",
    "completion_tokens_mean",
    "completion_tokens_p50",
    "reasoning_steps_mean",
    "total_gpu_seconds",
    "cost_per_correct_seconds",
    "cost_per_correct_seconds_ci95_low",
    "cost_per_correct_seconds_ci95_high",
    "cost_of_pass_seconds",
]

FAILURE_FIELDS = [
    "cell_id",
    "task",
    "model_family",
    "quant_config",
    "n",
    "parse_failure_rate",
    "empty_completion_rate",
    "truncation_rate",
    "invalid_answer_rate",
    "num_parse_failures",
    "num_empty_completions",
    "num_truncated",
    "num_invalid_answers",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_model_family(summary: dict[str, Any], metadata: dict[str, Any] | None) -> str:
    text = " ".join(
        str(x)
        for x in [
            summary.get("cell_id"),
            summary.get("model_path"),
            metadata.get("model_config", {}).get("model_id") if metadata else None,
        ]
        if x
    ).lower()
    if "qwen15b" in text or "qwen-1.5b" in text:
        return "Qwen-1.5B"
    if "qwen7b" in text or "qwen-7b" in text:
        return "Qwen-7B"
    if "llama8b" in text or "llama-8b" in text:
        return "Llama-8B"
    return "unknown"


def load_metadata_by_cell(archive: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted((archive / "metadata").glob("*.json")):
        data = load_json(path)
        out[data.get("cell_id", path.stem)] = data
    return out


def collect_rows(archive: Path) -> list[dict[str, Any]]:
    metadata = load_metadata_by_cell(archive)
    rows = []
    for path in sorted((archive / "results").glob("*_summary.json")):
        summary = load_json(path)
        cell_id = summary.get("cell_id") or path.name.removesuffix("_summary.json")
        meta = metadata.get(cell_id)
        model_cfg = meta.get("model_config", {}) if meta else {}
        row = dict(summary)
        row["cell_id"] = cell_id
        row["model_family"] = infer_model_family(summary, meta)
        row["model_id"] = model_cfg.get("model_id")
        row["model_config_path"] = meta.get("model_config_path") if meta else None
        row["task_config_path"] = meta.get("task_config_path") if meta else None
        row["decoding_config_path"] = meta.get("decoding_config_path") if meta else None
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="Output archive, e.g. outputs-hpc-...")
    parser.add_argument("--out-dir", default=None, help="Default: <archive>/paper_tables")
    args = parser.parse_args()

    archive = Path(args.archive)
    if not archive.is_absolute():
        archive = ROOT / archive
    out_dir = Path(args.out_dir) if args.out_dir else archive / "paper_tables"
    rows = collect_rows(archive)

    write_csv(out_dir / "paper_table_main.csv", rows, MAIN_FIELDS)
    write_csv(out_dir / "paper_table_efficiency.csv", rows, EFFICIENCY_FIELDS)
    write_csv(out_dir / "paper_table_failures.csv", rows, FAILURE_FIELDS)
    print(f"Wrote {len(rows)} rows to {out_dir}")


if __name__ == "__main__":
    main()
