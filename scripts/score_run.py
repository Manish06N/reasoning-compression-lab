#!/usr/bin/env python3
"""Score a raw JSONL run and write summary metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.correctness.scoring import get_scorer_metadata
from src.runners.artifact_validation import (
    ArtifactValidationError,
    validate_experiment_homogeneity,
    validate_experiment_homogeneity_warn,
)
from src.runners.multisample_validation import infer_n_samples, validate_multisample_groups
from src.runners.publication_mode import assert_clean_git_tree, is_publication_mode
from src.runners.scoring_pipeline import (
    attach_calibration,
    build_summary,
    load_raw_rows,
    score_all_rows,
    validate_raw_input,
    validate_scored_rows,
    validate_summary,
)
from src.schemas.validate import SchemaValidationError


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a raw inference JSONL file.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--require-calibration", action="store_true")
    parser.add_argument("--allow-parse-confidence-proxy", action="store_true")
    parser.add_argument(
        "--publication",
        action="store_true",
        help="Publication mode: full validation and fail-closed scoring.",
    )
    args = parser.parse_args()

    publication = is_publication_mode(cli_flag=args.publication)
    if publication:
        assert_clean_git_tree(ROOT)

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = ROOT / in_path
    stem = in_path.stem
    out_path = ROOT / (args.output or f"runs/scored/{stem}.jsonl")
    summary_path = ROOT / (args.summary or f"results/{stem}_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    validation = validate_raw_input(in_path, publication=publication)
    if not validation["valid"]:
        msg = f"raw input schema failed: {validation['errors'][:3]}"
        if publication:
            raise SchemaValidationError(msg)
        print(f"WARN: {msg}", file=sys.stderr)

    rows = load_raw_rows(in_path)
    if publication:
        try:
            validate_experiment_homogeneity(rows)
        except ArtifactValidationError as exc:
            raise SystemExit(f"ERROR: {exc}") from exc
        n_samples = infer_n_samples(rows)
        if n_samples and n_samples > 1:
            validate_multisample_groups(rows, n_samples=n_samples, publication_mode=True)
    else:
        for warning in validate_experiment_homogeneity_warn(rows):
            print(f"WARN: {warning}", file=sys.stderr)

    scored = score_all_rows(
        rows,
        allow_parse_proxy=args.allow_parse_confidence_proxy,
        publication=publication,
    )

    scored_validation = validate_scored_rows(scored, publication=publication)
    if not scored_validation["valid"]:
        msg = f"scored row schema failed: {scored_validation['errors'][:3]}"
        if publication:
            raise SchemaValidationError(msg)
        print(f"WARN: {msg}", file=sys.stderr)

    with out_path.open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = build_summary(
        scored, in_path=in_path, out_path=out_path, display_path=_display_path
    )
    summary.update(get_scorer_metadata(publication_mode=publication))
    attach_calibration(
        summary,
        scored,
        skip_calibration=args.skip_calibration,
        require_calibration=args.require_calibration,
        allow_parse_proxy=args.allow_parse_confidence_proxy,
    )

    summary_validation = validate_summary(summary, publication=publication)
    if not summary_validation["valid"]:
        msg = f"summary schema failed: {summary_validation['errors'][:3]}"
        if publication:
            raise SchemaValidationError(msg)
        print(f"WARN: {msg}", file=sys.stderr)

    if args.parquet:
        pq = Path(args.parquet)
        if not pq.is_absolute():
            pq = ROOT / pq
        import subprocess

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/export_parquet.py"),
                "--input",
                str(out_path),
                "--output",
                str(pq),
            ],
            check=True,
        )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Scored rows: {out_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
