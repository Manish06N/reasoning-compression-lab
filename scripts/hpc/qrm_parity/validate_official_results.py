#!/usr/bin/env python3
"""Validate an official-QRM result before promoting it to a larger run."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

WORD_RE = re.compile(r"\w+", re.UNICODE)


def _metric_is_correct(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) == 1.0
    return False


def _max_consecutive_word_run(text: str) -> int:
    words = [word.casefold() for word in WORD_RE.findall(text)]
    if not words:
        return 0
    longest = current = 1
    for previous, word in zip(words, words[1:]):
        current = current + 1 if word == previous else 1
        longest = max(longest, current)
    return longest


def analyze_rows(
    rows: list[dict[str, Any]],
    *,
    encode: Callable[[str], list[int]] | None = None,
    max_new_tokens: int = 32768,
    repetition_run_threshold: int = 20,
) -> dict[str, Any]:
    """Return aggregate and per-row health signals for official-QRM output."""
    details: list[dict[str, Any]] = []
    correct = boxed = metric_rows = token_limit_hits = repetition_rows = 0

    for index, row in enumerate(rows, start=1):
        text = row.get("generated_text")
        if not isinstance(text, str):
            raise ValueError(f"row {index} generated_text must be a string")

        metric = row.get("metrics", {}).get("extractive_match")
        has_metric = isinstance(metric, (bool, int, float))
        is_correct = _metric_is_correct(metric)
        has_boxed = "\\boxed" in text
        word_run = _max_consecutive_word_run(text)
        token_count = len(encode(text)) if encode is not None else None
        hit_token_limit = token_count is not None and token_count >= max_new_tokens
        has_repetition = word_run >= repetition_run_threshold

        metric_rows += int(has_metric)
        correct += int(is_correct)
        boxed += int(has_boxed)
        token_limit_hits += int(hit_token_limit)
        repetition_rows += int(has_repetition)
        details.append(
            {
                "row": index,
                "extractive_match": metric,
                "boxed": has_boxed,
                "completion_tokens": token_count,
                "hit_token_limit": hit_token_limit,
                "max_consecutive_word_run": word_run,
                "repetition_flag": has_repetition,
            }
        )

    count = len(rows)
    token_counts = [item["completion_tokens"] for item in details]
    numeric_tokens = [value for value in token_counts if value is not None]
    return {
        "rows": count,
        "metric_rows": metric_rows,
        "correct": correct,
        "accuracy": correct / count if count else 0.0,
        "boxed": boxed,
        "boxed_rate": boxed / count if count else 0.0,
        "token_limit_hits": token_limit_hits,
        "repetition_rows": repetition_rows,
        "completion_tokens_min": min(numeric_tokens) if numeric_tokens else None,
        "completion_tokens_max": max(numeric_tokens) if numeric_tokens else None,
        "completion_tokens_mean": (
            sum(numeric_tokens) / len(numeric_tokens) if numeric_tokens else None
        ),
        "details": details,
    }


def gate_errors(
    report: dict[str, Any],
    *,
    expected_rows: int,
    min_accuracy: float,
    min_boxed_rate: float,
    max_token_limit_hits: int,
    max_repetition_rows: int,
) -> list[str]:
    """Return human-readable gate failures."""
    errors = []
    if report["rows"] != expected_rows:
        errors.append(f"rows={report['rows']} (expected {expected_rows})")
    if report["metric_rows"] != report["rows"]:
        errors.append(
            f"numeric extractive_match metrics={report['metric_rows']}/{report['rows']}"
        )
    if report["accuracy"] < min_accuracy:
        errors.append(f"accuracy={report['accuracy']:.3f} (minimum {min_accuracy:.3f})")
    if report["boxed_rate"] < min_boxed_rate:
        errors.append(f"boxed_rate={report['boxed_rate']:.3f} (minimum {min_boxed_rate:.3f})")
    if report["token_limit_hits"] > max_token_limit_hits:
        errors.append(
            f"token_limit_hits={report['token_limit_hits']} (maximum {max_token_limit_hits})"
        )
    if report["repetition_rows"] > max_repetition_rows:
        errors.append(
            f"repetition_rows={report['repetition_rows']} (maximum {max_repetition_rows})"
        )
    return errors


def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("official-QRM result must be a JSON array")
    if not all(isinstance(row, dict) for row in data):
        raise ValueError("every official-QRM result row must be an object")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    parser.add_argument("--min-boxed-rate", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--max-token-limit-hits", type=int, default=0)
    parser.add_argument("--repetition-run-threshold", type=int, default=20)
    parser.add_argument("--max-repetition-rows", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if not 0.0 <= args.min_accuracy <= 1.0:
        parser.error("--min-accuracy must be between 0 and 1")
    if not 0.0 <= args.min_boxed_rate <= 1.0:
        parser.error("--min-boxed-rate must be between 0 and 1")

    encode = None
    if args.model is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

        def encode(text: str) -> list[int]:
            return tokenizer.encode(text, add_special_tokens=False)

    try:
        rows = _load_rows(args.result)
        report = analyze_rows(
            rows,
            encode=encode,
            max_new_tokens=args.max_new_tokens,
            repetition_run_threshold=args.repetition_run_threshold,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    errors = gate_errors(
        report,
        expected_rows=args.expected_rows,
        min_accuracy=args.min_accuracy,
        min_boxed_rate=args.min_boxed_rate,
        max_token_limit_hits=args.max_token_limit_hits,
        max_repetition_rows=args.max_repetition_rows,
    )
    report["result"] = str(args.result)
    report["model"] = str(args.model) if args.model is not None else None
    report["passed"] = not errors
    report["errors"] = errors

    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")

    if errors:
        print("FAIL: " + "; ".join(errors), file=sys.stderr)
        return 1
    print("PASS: official-QRM output gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
