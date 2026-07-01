"""Scoring helpers for math and majority-vote tasks."""

from __future__ import annotations

from collections import Counter
import random
import re
import statistics
from typing import Any, Dict, Optional, Sequence

from src.extraction.gpqa_extractor import extract_gpqa_letter, normalize_gpqa_letter
from src.extraction.math_extractor import (
    extract_boxed,
    extract_gold_answer,
    normalize_answer,
    normalize_completion_text,
)
from src.metrics.cost_per_correct import summarize_cost_metrics
from src.metrics.trace_length import summarize_trace_rows, trace_stats


def score_math_item(gold_solution: str, completion: str) -> Dict[str, Any]:
    completion = normalize_completion_text(completion)
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
        "boxed_answer_present": bool(pred),
        "answer_parse_success": bool(pred),
        **trace_stats(completion),
    }


def _extract_gsm8k_gold(answer: str) -> Optional[str]:
    if "####" in answer:
        return answer.rsplit("####", 1)[-1].strip()
    match = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", answer)
    return match[-1] if match else None


def _extract_numeric_answer(text: str) -> Optional[str]:
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    match = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return match[-1] if match else None


def score_gsm8k_item(gold_solution: str, completion: str) -> Dict[str, Any]:
    completion = normalize_completion_text(completion)
    pred = _extract_numeric_answer(completion)
    gold = _extract_gsm8k_gold(gold_solution)
    pred_norm = normalize_answer(pred).replace(",", "")
    gold_norm = normalize_answer(gold).replace(",", "")
    return {
        "pred_answer": pred,
        "gold_answer": gold,
        "correct": bool(pred_norm and gold_norm and pred_norm == gold_norm),
        "answer_parse_success": bool(pred),
        **trace_stats(completion),
    }


def score_gpqa_item(gold_letter: str, completion: str) -> Dict[str, Any]:
    completion = normalize_completion_text(completion)
    pred = extract_gpqa_letter(completion)
    pred_norm = normalize_gpqa_letter(pred)
    gold_norm = normalize_gpqa_letter(gold_letter)
    return {
        "pred_answer": pred,
        "gold_answer": gold_norm or None,
        "correct": bool(pred_norm and gold_norm and pred_norm == gold_norm),
        "answer_parse_success": bool(pred_norm),
        **trace_stats(completion),
    }


def score_item(row: dict) -> Dict[str, Any]:
    task = str(row.get("task", "math500")).lower()
    if task.startswith("gpqa"):
        return score_gpqa_item(str(row.get("gold_letter", "")), row.get("completion", ""))
    if task.startswith("gsm8k"):
        return score_gsm8k_item(row.get("gold_solution", ""), row.get("completion", ""))
    return score_math_item(row.get("gold_solution", ""), row.get("completion", ""))


def maj_at_k(correct_flags: Sequence[bool]) -> bool:
    """True if majority of samples are correct."""
    if not correct_flags:
        return False
    return sum(correct_flags) > len(correct_flags) / 2


def majority_vote_answer(answers: Sequence[str]) -> Optional[str]:
    if not answers:
        return None
    counts = Counter(str(a) for a in answers)
    winner, _ = counts.most_common(1)[0]
    return winner


def _try_math_verify(gold: Optional[str], pred: Optional[str]) -> Optional[bool]:
    if not gold or not pred:
        return None
    try:
        from math_verify import parse, verify
    except ImportError:
        return None
    try:
        return bool(verify(parse(gold), parse(pred)))
    except Exception:
        return None


def _mean(values: Sequence[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _percentile(values: Sequence[float], q: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic=None,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    """Percentile bootstrap confidence interval for a scalar statistic."""
    vals = list(values)
    stat = statistic or _mean
    if not vals:
        return {"value": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    observed = float(stat(vals))
    if len(vals) == 1:
        return {"value": observed, "ci_low": observed, "ci_high": observed}

    rng = random.Random(seed)
    n = len(vals)
    samples = []
    for _ in range(n_resamples):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        samples.append(float(stat(sample)))
    alpha = 1.0 - confidence
    return {
        "value": observed,
        "ci_low": _percentile(samples, alpha / 2.0),
        "ci_high": _percentile(samples, 1.0 - alpha / 2.0),
    }


def summarize_failure_rates(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {
            "parse_failure_rate": 0.0,
            "empty_completion_rate": 0.0,
            "truncation_rate": 0.0,
            "invalid_answer_rate": 0.0,
        }

    parse_failures = sum(1 for row in rows if not row.get("pred_answer"))
    empty = sum(1 for row in rows if not str(row.get("completion", "")).strip())
    trunc = sum(
        1
        for row in rows
        if row.get("truncated") is True
        or row.get("finish_reason") == "length"
        or (
            row.get("truncated") is None
            and row.get("completion_tokens") is not None
            and row["completion_tokens"]
            >= int((row.get("decoding_max_tokens") or row.get("max_tokens") or 32768) * 0.97)
        )
    )
    invalid = sum(1 for row in rows if not row.get("pred_answer") or row.get("correct") is None)
    return {
        "parse_failure_rate": parse_failures / n,
        "empty_completion_rate": empty / n,
        "truncation_rate": trunc / n,
        "invalid_answer_rate": invalid / n,
        "num_parse_failures": parse_failures,
        "num_empty_completions": empty,
        "num_truncated": trunc,
        "num_invalid_answers": invalid,
    }



def _summarize_numeric(rows: list[dict], field: str, *, include_max: bool = False) -> dict:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return {}
    out = {
        f"{field}_mean": sum(values) / len(values),
        f"{field}_p50": statistics.median(values),
    }
    if include_max:
        out[f"{field}_max"] = max(values)
    return out

def _cost_per_correct_stat(latencies_and_flags: Sequence[tuple[float, float]]) -> float:
    total_latency = sum(latency for latency, _ in latencies_and_flags)
    correct = sum(flag for _, flag in latencies_and_flags)
    if correct <= 0:
        return total_latency
    return total_latency / correct


def summarize_scored_rows(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "pass_at_1": 0.0}

    correct = sum(1 for row in rows if row.get("correct"))
    summary = {
        "n": n,
        "pass_at_1": correct / n,
        "num_correct": correct,
    }

    pass_ci = bootstrap_ci([1.0 if row.get("correct") else 0.0 for row in rows])
    summary["pass_at_1_ci95_low"] = pass_ci["ci_low"]
    summary["pass_at_1_ci95_high"] = pass_ci["ci_high"]

    latencies = [row["latency_sec"] for row in rows if row.get("latency_sec") is not None]
    vrams = [row["peak_vram_gb"] for row in rows if row.get("peak_vram_gb") is not None]
    prompt_tokens = [row["prompt_tokens"] for row in rows if row.get("prompt_tokens") is not None]
    completion_tokens = [row["completion_tokens"] for row in rows if row.get("completion_tokens") is not None]

    if latencies:
        summary["latency_sec_mean"] = sum(latencies) / len(latencies)
        summary["latency_sec_p50"] = statistics.median(latencies)
    if vrams:
        summary["peak_vram_gb_max"] = max(vrams)
        summary["peak_vram_gb_mean"] = sum(vrams) / len(vrams)
    if prompt_tokens:
        summary["prompt_tokens_mean"] = sum(prompt_tokens) / len(prompt_tokens)
    if completion_tokens:
        summary["completion_tokens_mean"] = sum(completion_tokens) / len(completion_tokens)
        summary["completion_tokens_p50"] = statistics.median(completion_tokens)

    summary.update(_summarize_numeric(rows, "time_to_first_token_sec"))
    summary.update(_summarize_numeric(rows, "tokens_per_second"))
    summary.update(_summarize_numeric(rows, "decode_tokens_per_second"))
    summary.update(_summarize_numeric(rows, "seconds_per_output_token"))
    summary.update(_summarize_numeric(rows, "vram_before_gb"))
    summary.update(_summarize_numeric(rows, "vram_after_gb"))
    summary.update(_summarize_numeric(rows, "vram_max_gb", include_max=True))
    summary.update(_summarize_numeric(rows, "gpu_util_mean"))
    summary.update(_summarize_numeric(rows, "gpu_util_max", include_max=True))
    summary.update(_summarize_numeric(rows, "power_watts_mean"))
    summary.update(_summarize_numeric(rows, "power_watts_max", include_max=True))
    summary.update(_summarize_numeric(rows, "tokens_per_joule"))

    energies = [row["energy_joules"] for row in rows if row.get("energy_joules") is not None]
    if energies:
        summary["energy_joules_total"] = sum(energies)
        summary["energy_joules_mean"] = sum(energies) / len(energies)
    finish_reasons = Counter(str(row.get("finish_reason")) for row in rows if row.get("finish_reason") is not None)
    if finish_reasons:
        summary["finish_reason_counts"] = dict(finish_reasons)

    summary.update(summarize_failure_rates(rows))
    summary.update(summarize_trace_rows(rows))
    summary.update(summarize_cost_metrics(rows))

    cpc_rows = [
        (float(row.get("latency_sec", 0.0)), 1.0 if row.get("correct") else 0.0)
        for row in rows
        if row.get("latency_sec") is not None
    ]
    if cpc_rows and any(flag for _, flag in cpc_rows):
        cpc_ci = bootstrap_ci(cpc_rows, statistic=_cost_per_correct_stat)
        summary["cost_per_correct_seconds_ci95_low"] = cpc_ci["ci_low"]
        summary["cost_per_correct_seconds_ci95_high"] = cpc_ci["ci_high"]
    return summary
