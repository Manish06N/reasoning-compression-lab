#!/usr/bin/env python3
"""Export answer-like fields from official-QRM campaign files (CPU only).

Official QRM ``inference.py`` writes a JSON *array* to a ``*.jsonl`` filename.
This exporter reads that format (and true JSONL) and writes sidecar files to
``--out``. It never writes into ``--jsonl-root``.

It does **not** parse ``\\boxed{}``, does **not** apply math-verify, and does
**not** treat ``metrics.extractive_match`` as answer agreement. That gold flag
is copied only as ``campaign_extractive_match`` for later scoring. Agreement
must follow docs/ANSWER_NORMALIZATION.md after recovery.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SEED_DIR_RE = re.compile(r"^(?P<model>.+)-seed(?P<seed>\d+)$")


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load a QRM result file: JSON array (usual) or newline-delimited JSON."""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path}: JSON started with '[' but is not a list")
        if not all(isinstance(row, dict) for row in data):
            raise ValueError(f"{path}: every row must be an object")
        return data
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: row must be an object")
        rows.append(row)
    return rows


def provenance_from_path(path: Path) -> dict[str, Any]:
    """Parse ``{checkpoint}-seed{N}/{DATASET}.jsonl`` from the campaign launcher."""
    match = SEED_DIR_RE.match(path.parent.name)
    model_dir = match.group("model") if match else path.parent.name
    seed = int(match.group("seed")) if match else None
    upper = model_dir.upper()
    if "AWQ" in upper:
        weight_format = "AWQ-4"
    elif "GPTQ" in upper:
        weight_format = "GPTQ-4"
    elif "FP8" in upper:
        weight_format = "FP8"
    else:
        weight_format = "BF16"
    if "Qwen" in model_dir:
        family = "Qwen-7B"
    elif "Llama" in model_dir:
        family = "Llama-8B"
    else:
        family = None
    return {
        "run_dir": path.parent.name,
        "dataset_file_stem": path.stem,
        "model_dir": model_dir,
        "seed": seed,
        "weight_format": weight_format,
        "model_family": family,
    }


def prediction_from_row(row: dict[str, Any]) -> str | None:
    """Copy an already-extracted answer string if the file stored one.

    Does not read ``gold`` and does not parse ``generated_text``.
    """
    metrics = row.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}
    for key in (
        "extracted_answer",
        "pred_answer",
        "predict",
        "prediction",
        "extractive_match_pred",
    ):
        value = row.get(key)
        if value in (None, ""):
            value = metrics.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
    return None


def problem_ref(row: dict[str, Any], index: int) -> dict[str, Any]:
    for key in ("id", "problem_id", "example_id", "doc_id", "uid"):
        value = row.get(key)
        if value not in (None, ""):
            return {"row": index, key: value}
    return {"row": index}


def sidecar_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    prompt = row.get("full_prompt")
    gold = row.get("gold")
    return {
        **problem_ref(row, index),
        "extracted_answer": prediction_from_row(row),
        "campaign_extractive_match": metrics.get("extractive_match"),
        "campaign_extractive_match_note": (
            "Gold-vs-extracted flag from the original evaluator. "
            "Not modal-answer agreement."
        ),
        "generated_text_present": isinstance(row.get("generated_text"), str),
        "full_prompt_prefix": prompt[:160] if isinstance(prompt, str) else None,
        "campaign_gold_present": gold not in (None, "", []),
    }


def peek(root: Path) -> int:
    paths = sorted(root.rglob("*.jsonl"))
    print(f"n_jsonl {len(paths)}")
    if not paths:
        print("no jsonl files", file=sys.stderr)
        return 2
    path = paths[0]
    rows = load_rows(path)
    print("file", path)
    print("n_rows_first_file", len(rows))
    print("load_format", "json_array_or_jsonl")
    if not rows:
        print("first file is empty")
        return 0
    row = rows[0]
    print("top_keys", sorted(row.keys()))
    metrics = row.get("metrics") or {}
    print("metrics_keys", sorted(metrics.keys()) if isinstance(metrics, dict) else type(metrics))
    for key in ("text", "generated_text", "completion", "extracted_answer", "pred", "prediction", "gold"):
        if key in row or (isinstance(metrics, dict) and key in metrics):
            print("present:", key)
    print("provenance", json.dumps(provenance_from_path(path)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="Sidecar directory (required unless --peek).")
    parser.add_argument(
        "--peek",
        action="store_true",
        help="Print first-file schema and exit without writing.",
    )
    args = parser.parse_args(argv)

    root = args.jsonl_root.resolve()
    if not root.is_dir():
        print(f"ERROR: jsonl root is not a directory: {root}", file=sys.stderr)
        return 2

    if args.peek:
        return peek(root)

    if args.out is None:
        parser.error("--out is required unless --peek")

    out = args.out.resolve()
    if out == root or root in out.parents:
        print(
            f"ERROR: --out {out} must not be inside --jsonl-root {root}",
            file=sys.stderr,
        )
        return 2

    out.mkdir(parents=True, exist_ok=True)
    n_files = n_rows = n_pred = 0
    for path in sorted(root.rglob("*.jsonl")):
        n_files += 1
        rows_in = load_rows(path)
        prov = provenance_from_path(path)
        records = []
        for index, row in enumerate(rows_in, start=1):
            rec = sidecar_record(row, index)
            rec.update(prov)
            rec["source"] = str(path)
            n_rows += 1
            n_pred += int(rec["extracted_answer"] is not None)
            records.append(rec)
        rel = path.relative_to(root)
        sidecar = out / (str(rel).replace("/", "__") + ".answers.json")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": str(path),
            "source_relative": str(rel),
            "provenance": prov,
            "note": (
                "Read-only export. campaign_extractive_match is gold correctness, "
                "not answer agreement. See docs/ANSWER_NORMALIZATION.md."
            ),
            "rows": records,
        }
        sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {sidecar} ({len(records)} rows)")
    print(f"files={n_files} rows={n_rows} with_extracted_answer={n_pred}")
    if n_files == 0:
        print("No *.jsonl files under", root, file=sys.stderr)
        return 2
    if n_pred == 0:
        print(
            "No extracted-answer strings in the files. Official QRM rows usually "
            "store generated_text + metrics.extractive_match only. Do not parse "
            "\\boxed{} here; use the frozen evaluator policy after rsync."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
