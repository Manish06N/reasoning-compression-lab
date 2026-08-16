#!/usr/bin/env python3
"""Recompute Paper 1 tables from released per-cell JSON.

Fixes the P0 analysis bugs called out in the 2026-08-16 reviews:
- pathology keys are ``token_limit_hits`` / ``repetition_rows`` (not truncation_count)
- near-cap rows are counted from completion_tokens (the cap detector never fires)
- pass@1 contrasts use problem-clustered bootstrap, not pooled Wilson / maj@5 McNemar
- token inflation uses all seeds, ratio-of-means *and* paired per-item deltas, stratified by correctness
- selective prediction is labeled as an oracle gold-hit diagnostic (no extracted answers in schema)

The compact validation JSON does not store extracted answer strings, so a deployable
modal-agreement gate cannot be recomputed from the public artifact.
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import statistics
from collections import defaultdict
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MATH_DIR = os.path.join(REPO_ROOT, "results", "math500")
GSM_DIR = os.path.join(REPO_ROOT, "results", "gsm8k")
GPQA_DIR = os.path.join(REPO_ROOT, "results", "gpqa")
REPORT_DIR = os.path.join(REPO_ROOT, "results", "reports")

MODELS = ["Qwen-7B", "Llama-8B"]
FORMATS = ["BF16", "FP8", "AWQ-4", "GPTQ-4"]
MATH_SEEDS = [42, 43, 44, 45, 46]
BREADTH_SEEDS = [42, 43, 44]
LOOP_THRESHOLD = 20
TOKEN_CAP = 32768
NEAR_CAP = 32500
N_BOOT = 10_000
BOOT_SEED = 0
EQUIV_MARGIN_PP = 1.0
GPU_USD_PER_HOUR = 1.50
TOK_PER_SEC = 65.0


def parse_cell(basename: str) -> tuple[str, str, int] | None:
    bn = basename.replace(".json", "")
    if "_math500_n500_seed" in bn:
        model_part, seed_s = bn.split("_math500_n500_seed")
    elif "_gsm8k_n1319_seed" in bn:
        model_part, seed_s = bn.split("_gsm8k_n1319_seed")
    elif "_gpqadiamond_n198_seed" in bn:
        model_part, seed_s = bn.split("_gpqadiamond_n198_seed")
    else:
        return None
    if "Qwen-7B" in model_part:
        model = "Qwen-7B"
    elif "Llama-8B" in model_part:
        model = "Llama-8B"
    else:
        return None
    if "-FP8" in model_part:
        fmt = "FP8"
    elif "-AWQ-4" in model_part:
        fmt = "AWQ-4"
    elif "-GPTQ-4" in model_part:
        fmt = "GPTQ-4"
    else:
        fmt = "BF16"
    return model, fmt, int(seed_s)


def load_dir(path: str) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    data: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for f in sorted(glob.glob(os.path.join(path, "*.json"))):
        parsed = parse_cell(os.path.basename(f))
        if parsed is None:
            continue
        model, fmt, seed = parsed
        with open(f) as fp:
            data[model][fmt][seed] = json.load(fp)
    return data


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - spread) * 100, min(1.0, center + spread) * 100


def n_choose_k(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cum = sum(n_choose_k(n, i) * (0.5**n) for i in range(k + 1))
    return min(1.0, 2.0 * cum)


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_means(values: list[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    out = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        out.append(s / n)
    return out


def bootstrap_mean_ci(item_means: list[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict[str, float]:
    boot = bootstrap_means(item_means, n_boot, seed)
    return {
        "mean": mean(item_means),
        "ci95_lo": percentile(boot, 2.5),
        "ci95_hi": percentile(boot, 97.5),
    }


def paired_delta_bootstrap(
    base: list[float],
    other: list[float],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, float]:
    """Clustered bootstrap of mean(other - base) by resampling items."""
    delta = [o - b for o, b in zip(other, base)]
    boot = bootstrap_means(delta, n_boot, seed)
    lo95, hi95 = percentile(boot, 2.5), percentile(boot, 97.5)
    lo90, hi90 = percentile(boot, 5.0), percentile(boot, 95.0)
    p_hi = sum(1 for x in boot if x >= 0.0) / len(boot)
    p_lo = sum(1 for x in boot if x <= 0.0) / len(boot)
    p = min(1.0, 2.0 * min(p_hi, p_lo))
    m = mean(delta)
    return {
        "delta": m,
        "delta_pp": m * 100,
        "ci95_lo_pp": lo95 * 100,
        "ci95_hi_pp": hi95 * 100,
        "ci90_lo_pp": lo90 * 100,
        "ci90_hi_pp": hi90 * 100,
        "p_value": p,
        "excludes_zero": (lo95 > 0) or (hi95 < 0),
        "tost_equiv_1pp": (lo90 * 100 >= -EQUIV_MARGIN_PP) and (hi90 * 100 <= EQUIV_MARGIN_PP),
    }


def item_correctness(cell_seeds: dict[int, dict[str, Any]], seeds: list[int], n_items: int) -> list[float]:
    """Per-item mean correctness across seeds."""
    acc = [0.0] * n_items
    for s in seeds:
        details = cell_seeds[s]["details"]
        for i in range(n_items):
            acc[i] += float(details[i].get("extractive_match", 0.0) == 1.0)
    k = float(len(seeds))
    return [a / k for a in acc]


def seed_accs(cell_seeds: dict[int, dict[str, Any]], seeds: list[int]) -> dict[int, float]:
    return {s: cell_seeds[s]["accuracy"] * 100 for s in seeds}


def pathology(cell_seeds: dict[int, dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    loops = 0
    cap_hits = 0
    near_cap = 0
    near_cap_unboxed = 0
    max_run = 0
    max_tok = 0
    flagged: list[dict[str, Any]] = []
    for s in seeds:
        d = cell_seeds[s]
        loops += int(d.get("repetition_rows", d.get("repetition_flag_count", 0)) or 0)
        cap_hits += int(d.get("token_limit_hits", d.get("hit_token_limit_count", 0)) or 0)
        for row in d["details"]:
            tok = int(row.get("completion_tokens") or 0)
            run = int(row.get("max_consecutive_word_run") or 0)
            max_tok = max(max_tok, tok)
            max_run = max(max_run, run)
            if tok >= NEAR_CAP:
                near_cap += 1
                if not row.get("boxed", True):
                    near_cap_unboxed += 1
            if row.get("repetition_flag"):
                flagged.append(
                    {
                        "seed": s,
                        "row": row.get("row"),
                        "completion_tokens": tok,
                        "max_consecutive_word_run": run,
                        "boxed": bool(row.get("boxed")),
                        "extractive_match": row.get("extractive_match"),
                    }
                )
    return {
        "loops": loops,
        "token_limit_hits": cap_hits,
        "near_cap": near_cap,
        "near_cap_unboxed": near_cap_unboxed,
        "max_completion_tokens": max_tok,
        "max_consecutive_word_run": max_run,
        "flagged_rows": flagged,
    }


def gold_hit_histogram(cell_seeds: dict[int, dict[str, Any]], seeds: list[int], n_items: int) -> dict[str, Any]:
    counts = [0] * (len(seeds) + 1)
    for i in range(n_items):
        c = sum(
            1
            for s in seeds
            if cell_seeds[s]["details"][i].get("extractive_match", 0.0) == 1.0
        )
        counts[c] += 1
    k = len(seeds)
    oracle = {}
    for th in range(1, k + 1):
        served = 0
        correct = 0
        for c, n in enumerate(counts):
            if c >= th or (k - c) >= th:
                served += n
                if c >= th:
                    correct += n
        oracle[f"{th}/{k}"] = {
            "coverage": 100.0 * served / n_items if n_items else 0.0,
            "selective_accuracy": 100.0 * correct / served if served else 0.0,
            "note": "oracle gold-hit gate; not answer-string agreement",
        }
    return {
        "gold_hit_counts": {str(i): counts[i] for i in range(k + 1)},
        "all_correct_unanimous": counts[k],
        "all_wrong": counts[0],
        "oracle_gate": oracle,
        "k3_coverage_tautological": k == 5,
    }


def token_strata(
    bf16_seeds: dict[int, dict[str, Any]],
    other_seeds: dict[int, dict[str, Any]],
    seeds: list[int],
    n_items: int,
) -> dict[str, Any]:
    """Paired (item, seed) token comparison stratified by correctness."""
    pairs = {"both_correct": [], "bf16_only": [], "quant_only": [], "both_wrong": []}
    all_bf16 = []
    all_other = []
    per_item_ratio = []
    per_item_delta = []
    for i in range(n_items):
        bf16_tok_item = []
        other_tok_item = []
        for s in seeds:
            b_row = bf16_seeds[s]["details"][i]
            o_row = other_seeds[s]["details"][i]
            bt = float(b_row.get("completion_tokens") or 0)
            ot = float(o_row.get("completion_tokens") or 0)
            bc = b_row.get("extractive_match", 0.0) == 1.0
            oc = o_row.get("extractive_match", 0.0) == 1.0
            all_bf16.append(bt)
            all_other.append(ot)
            bf16_tok_item.append(bt)
            other_tok_item.append(ot)
            rec = ot - bt
            if bc and oc:
                pairs["both_correct"].append(rec)
            elif bc and not oc:
                pairs["bf16_only"].append(rec)
            elif (not bc) and oc:
                pairs["quant_only"].append(rec)
            else:
                pairs["both_wrong"].append(rec)
        b_mean = sum(bf16_tok_item) / len(seeds)
        o_mean = sum(other_tok_item) / len(seeds)
        per_item_delta.append(o_mean - b_mean)
        if b_mean > 0:
            per_item_ratio.append(o_mean / b_mean)

    def summarize(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {"n": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0}
        return {
            "n": int(len(xs)),
            "mean": mean(xs),
            "median": percentile(xs, 50),
            "p90": percentile(xs, 90),
            "p95": percentile(xs, 95),
        }

    ratio_of_means = (sum(all_other) / len(all_other)) / (sum(all_bf16) / len(all_bf16)) - 1.0
    mean_of_ratios = (mean(per_item_ratio) - 1.0) if per_item_ratio else 0.0
    boot = bootstrap_means(per_item_delta)
    d_lo, d_hi = percentile(boot, 2.5), percentile(boot, 97.5)
    return {
        "ratio_of_means_pct": ratio_of_means * 100,
        "mean_of_per_item_ratios_pct": mean_of_ratios * 100,
        "mean_paired_delta_tokens": mean(per_item_delta),
        "median_paired_delta_tokens": percentile(per_item_delta, 50),
        "delta_ci95": [d_lo, d_hi],
        "iqr_delta": [percentile(per_item_delta, 25), percentile(per_item_delta, 75)],
        "p90_delta": percentile(per_item_delta, 90),
        "p95_delta": percentile(per_item_delta, 95),
        "strata": {k: summarize(v) for k, v in pairs.items()},
    }


def even_index_subset_demo(
    bf16_seeds: dict[int, dict[str, Any]],
    other_seeds: dict[int, dict[str, Any]],
) -> dict[str, float]:
    """Reproduce the old 200-item estimator artifact (seed 42, even indices, mean of ratios)."""
    idxs = list(range(0, 500, 2))[:200]
    ratios = []
    bf16_sum = other_sum = 0.0
    for i in idxs:
        bt = float(bf16_seeds[42]["details"][i].get("completion_tokens") or 0)
        ot = float(other_seeds[42]["details"][i].get("completion_tokens") or 0)
        bf16_sum += bt
        other_sum += ot
        if bt > 0:
            ratios.append(ot / bt)
    return {
        "mean_of_ratios_pct": (mean(ratios) - 1.0) * 100 if ratios else 0.0,
        "ratio_of_means_pct": (other_sum / bf16_sum - 1.0) * 100 if bf16_sum else 0.0,
    }


def cell_summary(cell_seeds: dict[int, dict[str, Any]], seeds: list[int], n_items: int) -> dict[str, Any]:
    accs = [cell_seeds[s]["accuracy"] * 100 for s in seeds]
    mean_acc = sum(accs) / len(accs)
    std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0
    pooled = sum(cell_seeds[s]["correct"] for s in seeds)
    item = item_correctness(cell_seeds, seeds, n_items)
    clustered = bootstrap_mean_ci(item)
    toks = [cell_seeds[s]["completion_tokens_mean"] for s in seeds]
    mean_tok = sum(toks) / len(toks)
    path = pathology(cell_seeds, seeds)
    w_lo, w_hi = wilson(pooled, n_items * len(seeds))
    return {
        "seed_accs": {str(s): accs[i] for i, s in enumerate(seeds)},
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "pooled_correct": pooled,
        "wilson_ci_95": [w_lo, w_hi],
        "clustered_acc_ci95": [clustered["ci95_lo"] * 100, clustered["ci95_hi"] * 100],
        "mean_tokens": mean_tok,
        **{k: path[k] for k in ("loops", "token_limit_hits", "near_cap", "near_cap_unboxed", "max_completion_tokens", "max_consecutive_word_run")},
        "flagged_rows": path["flagged_rows"],
    }


def holm(p_values: list[float], alpha: float = 0.05) -> list[dict[str, Any]]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    out = [{"p": p_values[i], "holm_alpha": None, "significant": False} for i in range(len(p_values))]
    m = len(p_values)
    for rank, i in enumerate(order):
        thresh = alpha / (m - rank)
        out[i]["holm_alpha"] = thresh
        out[i]["significant"] = p_values[i] < thresh
    return out


def analyze_benchmark(
    data: dict[str, dict[str, dict[int, dict[str, Any]]]],
    seeds: list[int],
    n_items: int,
    name: str,
) -> dict[str, Any]:
    summaries = {}
    for m in MODELS:
        for fmt in FORMATS:
            summaries[f"{m}_{fmt}"] = cell_summary(data[m][fmt], seeds, n_items)

    contrasts = []
    for m in MODELS:
        bf16_item = item_correctness(data[m]["BF16"], seeds, n_items)
        bf16_maj = None
        if len(seeds) == 5:
            bf16_counts = [0] * n_items
            for s in seeds:
                for i, row in enumerate(data[m]["BF16"][s]["details"]):
                    bf16_counts[i] += int(row.get("extractive_match", 0.0) == 1.0)
            bf16_maj = [c >= 3 for c in bf16_counts]
        for fmt in ["FP8", "AWQ-4", "GPTQ-4"]:
            other_item = item_correctness(data[m][fmt], seeds, n_items)
            boot = paired_delta_bootstrap(bf16_item, other_item)
            rec = {
                "contrast": f"{m} BF16 vs {fmt}",
                "model": m,
                "format": fmt,
                **boot,
            }
            if bf16_maj is not None:
                other_counts = [0] * n_items
                for s in seeds:
                    for i, row in enumerate(data[m][fmt][s]["details"]):
                        other_counts[i] += int(row.get("extractive_match", 0.0) == 1.0)
                other_maj = [c >= 3 for c in other_counts]
                n11 = n10 = n01 = n00 = 0
                for a, b in zip(bf16_maj, other_maj):
                    if a and b:
                        n11 += 1
                    elif a and not b:
                        n10 += 1
                    elif (not a) and b:
                        n01 += 1
                    else:
                        n00 += 1
                rec.update({"n11": n11, "n10": n10, "n01": n01, "n00": n00, "mcnemar_p": mcnemar_exact(n10, n01)})
            contrasts.append(rec)

    holm_pass1 = holm([c["p_value"] for c in contrasts])
    for c, h in zip(contrasts, holm_pass1):
        c["holm_alpha_pass1"] = h["holm_alpha"]
        c["holm_significant_pass1"] = h["significant"]
    if all("mcnemar_p" in c for c in contrasts):
        holm_mc = holm([c["mcnemar_p"] for c in contrasts])
        for c, h in zip(contrasts, holm_mc):
            c["holm_alpha_mcnemar"] = h["holm_alpha"]
            c["holm_significant_mcnemar"] = h["significant"]

    tokens = {}
    for m in MODELS:
        for fmt in ["FP8", "AWQ-4", "GPTQ-4"]:
            tokens[f"{m}_{fmt}"] = token_strata(data[m]["BF16"], data[m][fmt], seeds, n_items)
            if name == "math500":
                tokens[f"{m}_{fmt}"]["old_200_even_seed42"] = even_index_subset_demo(
                    data[m]["BF16"], data[m][fmt]
                )

    economics = {}
    gpu_per_sec = GPU_USD_PER_HOUR / 3600.0
    for key, s in summaries.items():
        pass1 = s["mean_acc"] / 100.0
        time_q = s["mean_tokens"] / TOK_PER_SEC
        cost_q = time_q * gpu_per_sec
        economics[key] = {
            "mean_tokens": s["mean_tokens"],
            "time_per_q_sec": time_q,
            "cost_per_q_dollars": cost_q,
            "c_pass_dollars": cost_q / pass1 if pass1 else None,
            "note": "token-implied Cpass at shared 65 tok/s; not measured wall-clock",
        }

    selective = {}
    if len(seeds) == 5:
        for m in MODELS:
            for fmt in FORMATS:
                selective[f"{m}_{fmt}"] = gold_hit_histogram(data[m][fmt], seeds, n_items)

    loop_rows = []
    for m in MODELS:
        for fmt in FORMATS:
            loop_rows.extend(
                [{**r, "model": m, "format": fmt} for r in summaries[f"{m}_{fmt}"]["flagged_rows"]]
            )

    return {
        "benchmark": name,
        "n_items": n_items,
        "seeds": seeds,
        "summary_statistics": summaries,
        "pass1_contrasts": contrasts,
        "token_analysis": tokens,
        "deployment_economics": economics,
        "selective_oracle": selective,
        "loop_rows": loop_rows,
        "total_loops": sum(summaries[k]["loops"] for k in summaries),
        "total_near_cap": sum(summaries[k]["near_cap"] for k in summaries),
        "total_token_limit_hits": sum(summaries[k]["token_limit_hits"] for k in summaries),
    }


def print_math(report: dict[str, Any]) -> None:
    print("\n" + "=" * 110)
    print("MATH-500 REANALYSIS (clustered CIs; real pathology keys; near-cap >= 32500)")
    print("=" * 110)
    print(
        f"{'Cell':<18} {'Mean±Std':<16} {'Clust. 95% CI':<18} {'Tok':>8} {'Loops':>5} {'Near':>5} {'MaxTok':>7}"
    )
    for m in MODELS:
        for fmt in FORMATS:
            s = report["summary_statistics"][f"{m}_{fmt}"]
            print(
                f"{m+' '+fmt:<18} {s['mean_acc']:6.2f}±{s['std_acc']:4.2f}  "
                f"[{s['clustered_acc_ci95'][0]:5.1f},{s['clustered_acc_ci95'][1]:5.1f}]  "
                f"{s['mean_tokens']:8.1f} {s['loops']:5d} {s['near_cap']:5d} {s['max_completion_tokens']:7d}"
            )
    print("\nPaired pass@1 clustered bootstrap vs BF16")
    print(f"{'Contrast':<28} {'Δpp':>7} {'95% CI':<18} {'p':>8} {'TOST±1pp':>9} {'McNemar p':>10}")
    for c in report["pass1_contrasts"]:
        print(
            f"{c['contrast']:<28} {c['delta_pp']:+6.2f}  "
            f"[{c['ci95_lo_pp']:+5.2f},{c['ci95_hi_pp']:+5.2f}]  "
            f"{c['p_value']:8.4f} {str(c['tost_equiv_1pp']):>9} {c.get('mcnemar_p', float('nan')):10.4f}"
        )
    print("\nToken inflation vs BF16 (all 5 seeds)")
    print(f"{'Contrast':<22} {'RoM%':>8} {'mean-ratio%':>12} {'Δtok':>8} {'old200 mean-ratio':>18} {'old200 RoM':>12}")
    for m in MODELS:
        for fmt in ["FP8", "AWQ-4", "GPTQ-4"]:
            t = report["token_analysis"][f"{m}_{fmt}"]
            old = t.get("old_200_even_seed42", {})
            print(
                f"{m+' '+fmt:<22} {t['ratio_of_means_pct']:+7.2f} {t['mean_of_per_item_ratios_pct']:+11.2f} "
                f"{t['mean_paired_delta_tokens']:+7.1f} {old.get('mean_of_ratios_pct', 0):+17.2f} "
                f"{old.get('ratio_of_means_pct', 0):+11.2f}"
            )
            strata = t["strata"]
            print(
                "    strata Δtok  "
                + "  ".join(
                    f"{k}={strata[k]['mean']:+.0f}(n={strata[k]['n']})"
                    for k in ("both_correct", "bf16_only", "quant_only", "both_wrong")
                )
            )
    print(f"\nTotal loops={report['total_loops']}  near-cap={report['total_near_cap']}  cap-hits={report['total_token_limit_hits']}")
    print(f"Loop threshold = {LOOP_THRESHOLD} consecutive identical words")


def phase5_compat(math_report: dict[str, Any]) -> dict[str, Any]:
    """Keep figure-generation keys, with corrected pathology and clustered CIs."""
    summary = {}
    for key, s in math_report["summary_statistics"].items():
        summary[key] = {
            "model": key.split("_")[0] + "_" + key.split("_")[1] if False else key.rsplit("_", 1)[0],
            "format": key.rsplit("_", 1)[1],
            "seed_accs": {int(k): v for k, v in s["seed_accs"].items()},
            "mean_acc": s["mean_acc"],
            "std_acc": s["std_acc"],
            "pooled_correct": s["pooled_correct"],
            "wilson_ci_95": tuple(s["wilson_ci_95"]),
            "clustered_acc_ci95": s["clustered_acc_ci95"],
            "mean_tokens": s["mean_tokens"],
            "truncations": s["token_limit_hits"],
            "loops": s["loops"],
            "near_cap": s["near_cap"],
        }
        # model key is Qwen-7B_BF16 → model Qwen-7B
        summary[key]["model"] = key[: key.rfind("_")]
    return {
        "dataset": "HuggingFaceH4/MATH-500",
        "sample_count_per_cell": 500,
        "total_evaluated_completions": 20000,
        "seeds": MATH_SEEDS,
        "loop_threshold": LOOP_THRESHOLD,
        "near_cap_threshold": NEAR_CAP,
        "summary_statistics": summary,
        "mcnemar_paired_contrasts": [
            {
                "contrast": c["contrast"],
                "model": c["model"],
                "format": c["format"],
                "n11": c.get("n11"),
                "n10": c.get("n10"),
                "n01": c.get("n01"),
                "n00": c.get("n00"),
                "p_value": c.get("mcnemar_p"),
                "holm_alpha": c.get("holm_alpha_mcnemar"),
                "is_significant": c.get("holm_significant_mcnemar"),
            }
            for c in math_report["pass1_contrasts"]
        ],
        "pass1_contrasts": math_report["pass1_contrasts"],
        "token_analysis": math_report["token_analysis"],
        "calibration_metrology": {},
        "deployment_economics": math_report["deployment_economics"],
        "note": "calibration_metrology omitted: gold-hit ECE is circular; see revision_reanalysis_report.json",
    }


def main() -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    math_data = load_dir(MATH_DIR)
    gsm_data = load_dir(GSM_DIR)
    gpqa_data = load_dir(GPQA_DIR)

    math_report = analyze_benchmark(math_data, MATH_SEEDS, 500, "math500")
    gsm_report = analyze_benchmark(gsm_data, BREADTH_SEEDS, 1319, "gsm8k")
    gpqa_report = analyze_benchmark(gpqa_data, BREADTH_SEEDS, 198, "gpqa")

    print_math(math_report)
    print("\n" + "=" * 110)
    print("GSM8K / GPQA pass@1 clustered bootstrap vs BF16")
    print("=" * 110)
    for bench, rep in [("GSM8K", gsm_report), ("GPQA", gpqa_report)]:
        print(f"\n{bench}")
        for c in rep["pass1_contrasts"]:
            print(
                f"  {c['contrast']:<28} {c['delta_pp']:+6.2f} pp  "
                f"[{c['ci95_lo_pp']:+5.2f},{c['ci95_hi_pp']:+5.2f}]  p={c['p_value']:.4f}  "
                f"Holm-sig={c['holm_significant_pass1']}"
            )
        print(
            f"  loops={rep['total_loops']}  near-cap={rep['total_near_cap']}  "
            f"cap-hits={rep['total_token_limit_hits']}"
        )

    master = {
        "meta": {
            "loop_threshold": LOOP_THRESHOLD,
            "token_cap": TOKEN_CAP,
            "near_cap_threshold": NEAR_CAP,
            "n_boot": N_BOOT,
            "equiv_margin_pp": EQUIV_MARGIN_PP,
            "extracted_answers_available": False,
            "schema_note": "Released JSON has extractive_match, completion_tokens, repetition_flag; no answer strings or finish_reason.",
        },
        "math500": math_report,
        "gsm8k": gsm_report,
        "gpqa_diamond": gpqa_report,
        "grid_totals": {
            "loops": math_report["total_loops"] + gsm_report["total_loops"] + gpqa_report["total_loops"],
            "near_cap": math_report["total_near_cap"] + gsm_report["total_near_cap"] + gpqa_report["total_near_cap"],
            "token_limit_hits": (
                math_report["total_token_limit_hits"]
                + gsm_report["total_token_limit_hits"]
                + gpqa_report["total_token_limit_hits"]
            ),
        },
    }
    out_rev = os.path.join(REPORT_DIR, "revision_reanalysis_report.json")
    with open(out_rev, "w") as fp:
        json.dump(master, fp, indent=2)
        fp.write("\n")
    print(f"\nWrote {out_rev}")

    phase5 = phase5_compat(math_report)
    out_p5 = os.path.join(REPORT_DIR, "phase5_statistical_analysis_report.json")
    with open(out_p5, "w") as fp:
        json.dump(phase5, fp, indent=2)
        fp.write("\n")

    multitask = {"math500": {}, "gsm8k": {}, "gpqa_diamond": {}}
    for label, rep, key in (
        ("math500", math_report, "math500"),
        ("gsm8k", gsm_report, "gsm8k"),
        ("gpqa_diamond", gpqa_report, "gpqa_diamond"),
    ):
        for cell, s in rep["summary_statistics"].items():
            multitask[key][cell] = {
                "model": cell[: cell.rfind("_")],
                "format": cell.rsplit("_", 1)[1],
                "accuracy_mean": s["mean_acc"],
                "accuracy_std": s["std_acc"],
                "mean_tokens": s["mean_tokens"],
                "truncations": s["token_limit_hits"],
                "loops": s["loops"],
                "near_cap": s["near_cap"],
                "clustered_acc_ci95": s["clustered_acc_ci95"],
            }
    with open(os.path.join(REPORT_DIR, "multitask_benchmark_summary.json"), "w") as fp:
        json.dump(multitask, fp, indent=2)
        fp.write("\n")

    with open(os.path.join(REPORT_DIR, "selective_prediction_report.json"), "w") as fp:
        json.dump(
            {
                "definition": "Oracle gold-hit gate: serve if #gold-correct >= k OR #gold-wrong >= k. Not deployable. Compact JSON has no extracted answers.",
                "math500": math_report["selective_oracle"],
            },
            fp,
            indent=2,
        )
        fp.write("\n")

    token_audit = {
        "definition": "Full MATH-500 grid, all 5 seeds, paired vs BF16. Not a 200-item even-index subset.",
        "analysis": math_report["token_analysis"],
    }
    with open(os.path.join(REPORT_DIR, "trace_audit_report.json"), "w") as fp:
        json.dump(token_audit, fp, indent=2)
        fp.write("\n")

    print("Updated phase5 / multitask / selective / token reports.")
    print(
        f"GRID TOTALS  loops={master['grid_totals']['loops']}  "
        f"near-cap={master['grid_totals']['near_cap']}  "
        f"cap-hits={master['grid_totals']['token_limit_hits']}"
    )


if __name__ == "__main__":
    main()
