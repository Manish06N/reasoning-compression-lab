"""Scoring helpers for math tasks."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.extraction.math_extractor import extract_boxed, extract_gold_answer, normalize_answer


def score_math_item(gold_solution: str, completion: str) -> Dict[str, Any]:
    pred = extract_boxed(completion)
    gold = extract_gold_answer(gold_solution)
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    correct = bool(pred_norm and gold_norm and pred_norm == gold_norm)
    math_verify_correct = _try_math_verify(gold, pred)

    if math_verify_correct is not None:
        correct = math_verify_correct

    return {
        "pred_answer": pred,
        "gold_answer": gold,
        "correct": correct,
    }


def _try_math_verify(gold: Optional[str], pred: Optional[str]) -> Optional[bool]:
    if not gold or not pred:
        return None
    try:
        from math_verify import parse, verify
    except ImportError:
        return None
    try:
        gold_parsed = parse(gold)
        pred_parsed = parse(pred)
        return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        return None


def summarize_scored_rows(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "pass_at_1": 0.0}
    correct = sum(1 for row in rows if row.get("correct"))
    latencies = [row["latency_sec"] for row in rows if row.get("latency_sec") is not None]
    vrams = [row["peak_vram_gb"] for row in rows if row.get("peak_vram_gb") is not None]
    prompt_tokens = [row["prompt_tokens"] for row in rows if row.get("prompt_tokens") is not None]
    completion_tokens = [row["completion_tokens"] for row in rows if row.get("completion_tokens") is not None]

    summary = {
        "n": n,
        "pass_at_1": correct / n,
        "num_correct": correct,
    }
    if latencies:
        summary["latency_sec_mean"] = sum(latencies) / len(latencies)
        summary["latency_sec_p50"] = sorted(latencies)[len(latencies) // 2]
    if vrams:
        summary["peak_vram_gb_max"] = max(vrams)
        summary["peak_vram_gb_mean"] = sum(vrams) / len(vrams)
    if prompt_tokens:
        summary["prompt_tokens_mean"] = sum(prompt_tokens) / len(prompt_tokens)
    if completion_tokens:
        summary["completion_tokens_mean"] = sum(completion_tokens) / len(completion_tokens)
    return summary
