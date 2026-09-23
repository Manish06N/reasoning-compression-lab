#!/usr/bin/env python3
"""CPU sensitivities added in the September 2026 revision.

Reads only released artifacts (compact per-cell JSON and the recovered MATH-500
answer strings) and never rewrites frozen reports. Produces
``results/reports/revision_sensitivities.json`` with:

- ``gpqa_borderline``: the GPQA-Diamond Qwen AWQ-4 contrast rerun with
  B=10^6, its spread over 20 bootstrap seeds at B=10^4, an item-level
  sign-flip permutation test, a paired t-test, and the Holm-6 / Holm-18
  adjustments that follow from each.
- ``seed_placebo``: the BF16-correct length estimator applied to BF16 seed s
  versus BF16 seed t != s (20 ordered pairs), next to the seed-matched
  quantized contrasts, with item-level bootstrap CIs.
- ``k_curve``: k-sample unanimity (k=2..5) on MATH-500, averaged over all
  seed subsets, against a one-sample shortest-trace rule at matched coverage.
- ``near_cap``: completion-length bands used as the truncation proxy.
- ``item_order``: how often a correct recovered prediction equals the public
  MATH-500 gold at the same row position (evidence that compact rows are not
  in public release order).

Usage:
    python scripts/analysis/revision_sensitivities.py --write
    python scripts/analysis/revision_sensitivities.py --check
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
RECOVERED = RESULTS / "recovered" / "math500_modal_inputs.jsonl"
REPORT = RESULTS / "reports" / "revision_reanalysis_report.json"
OUT = RESULTS / "reports" / "revision_sensitivities.json"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"

FAMILIES = ("Qwen-7B", "Llama-8B")
FORMATS = ("BF16", "FP8", "AWQ-4", "GPTQ-4")
BENCH = {
    "math500": ("math500", range(42, 47), 500),
    "gsm8k": ("gsm8k", range(42, 45), 1319),
    "gpqa": ("gpqadiamond", range(42, 45), 198),
}
B = 10_000


def load_cell(bench: str, family: str, fmt: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (correct, tokens) arrays of shape (seeds, items) in row order."""
    tag, seeds, n = BENCH[bench]
    stem = f"DeepSeek-R1-Distill-{family}" + ("" if fmt == "BF16" else f"-{fmt}")
    acc, tok = [], []
    for seed in seeds:
        path = RESULTS / bench / f"{stem}_{tag}_n{n}_seed{seed}.json"
        rows = sorted(json.loads(path.read_text())["details"], key=lambda r: r["row"])
        assert len(rows) == n, path
        acc.append([r["extractive_match"] for r in rows])
        tok.append([r["completion_tokens"] for r in rows])
    return np.array(acc, dtype=float), np.array(tok, dtype=float)


def holm_adjusted(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted


def gpqa_borderline() -> dict:
    base, _ = load_cell("gpqa", "Qwen-7B", "BF16")
    quant, _ = load_cell("gpqa", "Qwen-7B", "AWQ-4")
    d = quant.mean(0) - base.mean(0)
    n = len(d)

    def boot_p(rng: np.random.Generator, reps: int, chunk: int = 20_000) -> float:
        hi = lo = 0
        done = 0
        while done < reps:
            size = min(chunk, reps - done)
            means = d[rng.integers(0, n, (size, n))].mean(1)
            hi += int((means >= 0).sum())
            lo += int((means <= 0).sum())
            done += size
        return min(1.0, 2 * min(hi, lo) / reps)

    spread = [boot_p(np.random.default_rng(s), B) for s in range(20)]
    p_1e6 = boot_p(np.random.default_rng(123), 1_000_000)
    signs = np.random.default_rng(1).choice([-1, 1], (200_000, n))
    perm = (signs * d).mean(1)
    p_perm = float((np.abs(perm) >= abs(d.mean()) - 1e-12).mean())
    p_t = float(stats.ttest_1samp(d, 0).pvalue)

    report = json.loads(REPORT.read_text())
    gpqa = report["gpqa_diamond"]["pass1_contrasts"]
    others6 = [c["p_value"] for c in gpqa if not (c["model"] == "Qwen-7B" and c["format"] == "AWQ-4")]
    others18 = [
        c["p_value"]
        for bench in ("math500", "gsm8k", "gpqa_diamond")
        for c in report[bench]["pass1_contrasts"]
        if not (bench == "gpqa_diamond" and c["model"] == "Qwen-7B" and c["format"] == "AWQ-4")
    ]

    def adj(p: float, others: list[float]) -> float:
        return holm_adjusted([p] + others)[0]

    return {
        "delta_pp": 100 * float(d.mean()),
        "p_bootstrap_B1e4_seed_spread": {"min": min(spread), "median": float(np.median(spread)), "max": max(spread),
                                          "holm6_significant_count": sum(adj(p, others6) < 0.05 for p in spread),
                                          "n_seeds": len(spread)},
        "p_bootstrap_B1e6": p_1e6,
        "p_signflip_permutation": p_perm,
        "p_paired_t": p_t,
        "holm6_adjusted": {"bootstrap_B1e6": adj(p_1e6, others6), "permutation": adj(p_perm, others6),
                           "paired_t": adj(p_t, others6)},
        "holm18_adjusted_bootstrap_B1e6": adj(p_1e6, others18),
        "label": "borderline",
    }


def seed_placebo() -> dict:
    def pooled(a_ref, t_ref, a_oth, t_oth, pairs):
        n = a_ref.shape[1]
        s_ref, n_ref, s_oth, n_oth = (np.zeros(n) for _ in range(4))
        for r, o in pairs:
            delta = t_oth[o] - t_ref[r]
            s_ref += delta * a_ref[r]
            n_ref += a_ref[r]
            s_oth += delta * a_oth[o]
            n_oth += a_oth[o]
        idx = np.random.default_rng(0).integers(0, n, (B, n))
        b_ref = s_ref[idx].sum(1) / n_ref[idx].sum(1)
        b_oth = s_oth[idx].sum(1) / n_oth[idx].sum(1)
        return {
            "reference_correct": float(s_ref.sum() / n_ref.sum()),
            "reference_correct_ci95": [float(x) for x in np.percentile(b_ref, [2.5, 97.5])],
            "other_correct": float(s_oth.sum() / n_oth.sum()),
            "other_correct_ci95": [float(x) for x in np.percentile(b_oth, [2.5, 97.5])],
        }

    out: dict = {}
    for family in FAMILIES:
        a, t = load_cell("math500", family, "BF16")
        out[f"{family}_BF16_placebo"] = pooled(a, t, a, t, list(itertools.permutations(range(5), 2)))
        for fmt in FORMATS[1:]:
            aq, tq = load_cell("math500", family, fmt)
            rec = pooled(a, t, aq, tq, [(s, s) for s in range(5)])
            same = float(np.mean([(a[s] == aq[s]).mean() for s in range(5)]))
            cross = float(np.mean([(a[s] == aq[u]).mean() for s in range(5) for u in range(5) if s != u]))
            rec["correctness_agreement_same_seed"] = same
            rec["correctness_agreement_cross_seed"] = cross
            out[f"{family}_{fmt}"] = rec
    return out


def k_curve() -> dict:
    import sympy
    from math_verify import verify

    @functools.lru_cache(maxsize=None)
    def parse(text: str):
        try:
            return sympy.sympify(text, evaluate=True)
        except Exception:
            return None

    @functools.lru_cache(maxsize=None)
    def equivalent(x: str, y: str) -> bool:
        if x == y:
            return True
        px, py = parse(x), parse(y)
        if px is None or py is None:
            return False
        try:
            return bool(verify(px, py) and verify(py, px))
        except Exception:
            return False

    preds: dict = defaultdict(dict)
    with RECOVERED.open() as handle:
        for line in handle:
            r = json.loads(line)
            key = r["extracted_pred_repr"][0] if r["extracted_pred_repr"] else None
            preds[(r["model"], r["format"])][(r["problem_index"], r["seed"])] = (key, r["campaign_extractive_match"])

    def modal(rows):
        parent = list(range(len(rows)))

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if rows[i][0] is not None and rows[j][0] is not None and equivalent(rows[i][0], rows[j][0]):
                    parent[find(i)] = find(j)
        groups: dict = defaultdict(list)
        for i in range(len(rows)):
            groups[find(i)].append(i)
        ordered = sorted(groups.values(), key=len, reverse=True)
        if len(ordered) > 1 and len(ordered[0]) == len(ordered[1]):
            return 0, 0.0
        return len(ordered[0]), rows[ordered[0][0]][1]

    seeds = list(range(42, 47))
    out: dict = {}
    for family in FAMILIES:
        for fmt in FORMATS:
            cell = preds[(family, fmt)]
            acc, tok = load_cell("math500", family, fmt)
            rec = {}
            for k in (2, 3, 4, 5):
                covs, risks = [], []
                for subset in itertools.combinations(seeds, k):
                    served = wrong = 0
                    for i in range(500):
                        size, correct = modal([cell[(i, s)] for s in subset])
                        if size == k:
                            served += 1
                            wrong += correct == 0
                    covs.append(served / 500)
                    risks.append(wrong / served)
                coverage = float(np.mean(covs))
                keep = int(round(coverage * 500))
                length_risk = float(np.mean([1 - acc[s][np.argsort(tok[s], kind="stable")[:keep]].mean() for s in range(5)]))
                rec[str(k)] = {"coverage_pct": 100 * coverage, "risk_pct": 100 * float(np.mean(risks)),
                               "length_rule_risk_pct": 100 * length_risk, "tokens_per_item": k * float(tok.mean())}
            out[f"{family}_{fmt}"] = rec
    return out


def near_cap() -> dict:
    out = {}
    for bench in BENCH:
        tokens = np.concatenate([load_cell(bench, f, m)[1].ravel() for f in FAMILIES for m in FORMATS])
        out[bench] = {
            "ge_32500": int((tokens >= 32_500).sum()),
            "band_32600_32739": int(((tokens >= 32_600) & (tokens <= 32_739)).sum()),
            "band_31957_32499": int(((tokens >= 31_957) & (tokens < 32_500)).sum()),
            "max": int(tokens.max()),
        }
    return out


def item_order() -> dict:
    try:
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceH4/MATH-500", split="test", revision=MATH500_REVISION)
    except Exception as exc:  # offline without cache
        return {"skipped": f"MATH-500 unavailable: {type(exc).__name__}"}
    gold = [row["answer"].strip() for row in ds]
    match = total = 0
    with RECOVERED.open() as handle:
        for line in handle:
            r = json.loads(line)
            g = gold[r["problem_index"]]
            if r["campaign_extractive_match"] == 1.0 and r["extracted_pred_repr"] and re.fullmatch(r"-?\d+", g):
                total += 1
                match += r["extracted_pred_repr"][0] == g
    return {"correct_integer_gold_rows": total, "pred_equals_public_gold_same_position": match}


def compute() -> dict:
    return {
        "gpqa_borderline": gpqa_borderline(),
        "seed_placebo": seed_placebo(),
        "near_cap": near_cap(),
        "item_order": item_order(),
        "k_curve": k_curve(),
    }


def compare(expected, actual, path="", tol=1e-9) -> list[str]:
    errors = []
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key not in actual:
                errors.append(f"{path}/{key}: missing")
            else:
                errors += compare(value, actual[key], f"{path}/{key}", tol)
    elif isinstance(expected, list):
        for i, (e, a) in enumerate(zip(expected, actual)):
            errors += compare(e, a, f"{path}[{i}]", tol)
    elif isinstance(expected, float):
        if abs(expected - actual) > tol * max(1.0, abs(expected)):
            errors.append(f"{path}: expected {expected} got {actual}")
    elif expected != actual:
        errors.append(f"{path}: expected {expected!r} got {actual!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="recompute and write the JSON report")
    mode.add_argument("--check", action="store_true", help="recompute and compare with the JSON report")
    args = parser.parse_args()
    result = compute()
    if args.write:
        OUT.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {OUT.relative_to(ROOT)}")
        return 0
    stored = json.loads(OUT.read_text())
    if "skipped" in result["item_order"]:
        print(f"note: item_order not checked ({result['item_order']['skipped']})")
        stored.pop("item_order", None)
        result.pop("item_order")
    errors = compare(stored, result)
    if errors:
        print("FAIL: revision sensitivities differ from stored report", file=sys.stderr)
        for err in errors[:30]:
            print(f"  {err}", file=sys.stderr)
        return 1
    print(f"PASS: {OUT.relative_to(ROOT)} reproduces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
