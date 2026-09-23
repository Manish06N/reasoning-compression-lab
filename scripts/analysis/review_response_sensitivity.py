#!/usr/bin/env python3
"""CPU sensitivities requested after the manuscript review.

Does not change frozen pass@1, token, or serving JSON. Prints:
- quant-correct length deltas (n-weighted Both-OK and Quant-only)
- campaign-length seconds per correct answer
- rank-1 frequencies for pass@1 and for campaign-length cost
- one-sample length abstention at the published 5/5 coverage
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results/reports/revision_reanalysis_report.json"
JSONL = ROOT / "results/recovered/math500_modal_inputs.jsonl"

TOKS = {
    ("Qwen-7B", "BF16"): (43.92, 252.72),
    ("Qwen-7B", "FP8"): (62.50, 449.79),
    ("Qwen-7B", "AWQ-4"): (76.89, 418.15),
    ("Qwen-7B", "GPTQ-4"): (82.67, 488.26),
    ("Llama-8B", "BF16"): (72.34, 366.98),
    ("Llama-8B", "FP8"): (78.75, 481.42),
    ("Llama-8B", "AWQ-4"): (70.20, 391.11),
    ("Llama-8B", "GPTQ-4"): (63.49, 366.16),
}
FMTS = ["BF16", "FP8", "AWQ-4", "GPTQ-4"]
SEEDS = (42, 43, 44, 45, 46)
COV5 = {
    ("Qwen-7B", "BF16"): 0.884,
    ("Qwen-7B", "FP8"): 0.888,
    ("Qwen-7B", "AWQ-4"): 0.864,
    ("Qwen-7B", "GPTQ-4"): 0.868,
    ("Llama-8B", "BF16"): 0.762,
    ("Llama-8B", "FP8"): 0.782,
    ("Llama-8B", "AWQ-4"): 0.702,
    ("Llama-8B", "GPTQ-4"): 0.754,
}


def load_preds() -> dict:
    preds: dict = defaultdict(dict)
    with JSONL.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["model"], row["format"], row["problem_index"])
            preds[key][row["seed"]] = (
                row["campaign_extractive_match"],
                row["completion_tokens"],
            )
    return preds


def quant_correct() -> None:
    report = json.loads(REPORT.read_text())
    print("quant-correct conditional (Both-OK ∪ Quant-only)")
    for key, rec in report["math500"]["token_analysis"].items():
        both = rec["strata"]["both_correct"]
        quant = rec["strata"]["quant_only"]
        n = both["n"] + quant["n"]
        mean = (both["n"] * both["mean"] + quant["n"] * quant["mean"]) / n
        bf = rec["lian_bf16_correct_delta"]["mean"]
        print(f"  {key}: n={n} quant-correct={mean:.1f} bf16-correct={bf:.1f}")


def campaign_seconds() -> None:
    report = json.loads(REPORT.read_text())
    stats = report["math500"]["summary_statistics"]
    print("campaign seconds per correct = mean tokens / tok/s / pass@1")
    for fam in ("Qwen-7B", "Llama-8B"):
        for fmt in FMTS:
            cell = stats[f"{fam}_{fmt}"]
            tokens = cell["mean_tokens"]
            acc = cell["mean_acc"] / 100.0
            a_tps, b_tps = TOKS[(fam, fmt)]
            print(
                f"  {fam} {fmt}: A={tokens / a_tps / acc:.2f} "
                f"B={tokens / b_tps / acc:.2f}"
            )


def length_abstention(preds: dict) -> None:
    print("one-sample shortest-k risk at published 5/5 coverage")
    for fam in ("Qwen-7B", "Llama-8B"):
        for fmt in FMTS:
            k = round(COV5[(fam, fmt)] * 500)
            risks = []
            for seed in SEEDS:
                pairs = []
                for index in range(500):
                    correct, tokens = preds[(fam, fmt, index)][seed]
                    pairs.append((tokens, correct))
                pairs.sort()
                wrong = sum(1 for _, correct in pairs[:k] if correct < 1)
                risks.append(wrong / k)
            mean = sum(risks) / len(risks)
            print(
                f"  {fam} {fmt}: k={k} mean_risk={100 * mean:.2f}% "
                f"range={100 * min(risks):.2f}-{100 * max(risks):.2f}"
            )


def kendall(order_a: list[str], order_b: list[str]) -> float:
    rank_b = {fmt: i for i, fmt in enumerate(order_b)}
    concordant = discordant = 0
    for i, left in enumerate(order_a):
        for right in order_a[i + 1 :]:
            sign = (order_a.index(right) - order_a.index(left)) * (
                rank_b[right] - rank_b[left]
            )
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    pairs = len(order_a) * (len(order_a) - 1) / 2
    return (concordant - discordant) / pairs


def ranks(preds: dict, replicates: int = 10000) -> None:
    acc: dict = defaultdict(dict)
    tokens: dict = defaultdict(dict)
    for (fam, fmt, index), seeds in preds.items():
        acc[(fam, fmt)][index] = sum(seeds[s][0] for s in SEEDS) / 5
        tokens[(fam, fmt)][index] = sum(seeds[s][1] for s in SEEDS) / 5
    rng = random.Random(0)
    indexes = list(range(500))
    print(f"rank-1 frequency, item bootstrap B={replicates}, seed 0, tok/s fixed")
    for fam in ("Qwen-7B", "Llama-8B"):
        def cell_stats(sample: list[int]) -> dict:
            out = {}
            n = len(sample)
            for fmt in FMTS:
                mean_acc = sum(acc[(fam, fmt)][i] for i in sample) / n
                mean_tokens = sum(tokens[(fam, fmt)][i] for i in sample) / n
                a_tps, b_tps = TOKS[(fam, fmt)]
                out[fmt] = (
                    mean_acc,
                    mean_tokens / a_tps / mean_acc,
                    mean_tokens / b_tps / mean_acc,
                )
            return out

        point = cell_stats(indexes)
        orders = {}
        for name, getter, higher_better in (
            ("pass@1", lambda v: v[0], True),
            ("lenA", lambda v: v[1], False),
            ("lenB", lambda v: v[2], False),
        ):
            orders[name] = sorted(
                FMTS, key=lambda fmt: getter(point[fmt]), reverse=higher_better
            )
            pretty = ", ".join(
                f"{fmt} {getter(point[fmt]):.2f}"
                if name != "pass@1"
                else f"{fmt} {100 * getter(point[fmt]):.2f}"
                for fmt in orders[name]
            )
            print(f"  {fam} {name}: {pretty}")
        wins = {name: {fmt: 0.0 for fmt in FMTS} for name in orders}
        for _ in range(replicates):
            sample = [indexes[rng.randrange(500)] for _ in range(500)]
            stats = cell_stats(sample)
            for name, getter, higher_better in (
                ("pass@1", lambda v: v[0], True),
                ("lenA", lambda v: v[1], False),
                ("lenB", lambda v: v[2], False),
            ):
                values = {fmt: getter(stats[fmt]) for fmt in FMTS}
                best = max(values.values()) if higher_better else min(values.values())
                champs = [fmt for fmt, value in values.items() if abs(value - best) < 1e-12]
                share = 1.0 / len(champs)
                for fmt in champs:
                    wins[name][fmt] += share
        for name in orders:
            freqs = " ".join(f"{fmt}={wins[name][fmt] / replicates:.3f}" for fmt in FMTS)
            print(f"  {fam} {name} P(rank 1): {freqs}")
        print(
            f"  {fam} Kendall pass@1 vs lenA: "
            f"{kendall(orders['pass@1'], orders['lenA']):.2f}; "
            f"lenA vs lenB: {kendall(orders['lenA'], orders['lenB']):.2f}"
        )


def main() -> None:
    preds = load_preds()
    quant_correct()
    campaign_seconds()
    length_abstention(preds)
    ranks(preds)


if __name__ == "__main__":
    main()
