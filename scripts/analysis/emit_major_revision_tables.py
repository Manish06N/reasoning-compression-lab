#!/usr/bin/env python3
"""Independently recompute major-revision tables from raw JSON, then freeze markdown.

Does not rewrite LaTeX. Stdlib only. Run after the canonical analysis scripts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

import modal_agreement_analysis as modal  # noqa: E402
import revision_reanalysis as rev  # noqa: E402

REPORT = os.path.join(REPO_ROOT, "results", "reports", "revision_reanalysis_report.json")
SERVING = os.path.join(
    REPO_ROOT,
    "results",
    "reports",
    "measured_serving_confirmation",
    "measured_serving_confirmation_report.json",
)
MODAL = os.path.join(REPO_ROOT, "results", "reports", "modal_agreement_report.json")
OUT_MD = os.path.join(REPO_ROOT, "results", "reports", "major_revision_tables.md")
OUT_VALID = os.path.join(REPO_ROOT, "results", "reports", "major_revision_validation.md")
GPU_USD_PER_SEC = 1.50 / 3600.0
STRATA = ("both_correct", "bf16_only", "quant_only", "both_wrong")


def _fmt_pp(x: float) -> str:
    return f"{x:+.2f}"


def _ci95(lo: float, hi: float) -> str:
    return f"[{lo:+.2f}, {hi:+.2f}]"


def independent_stratum_counts(
    bf16_seeds: dict[int, dict[str, Any]],
    other_seeds: dict[int, dict[str, Any]],
    seeds: list[int],
    n_items: int,
) -> dict[str, dict[str, float]]:
    """Recompute 2×2 n and mean Δ without calling token_strata()."""
    buckets: dict[str, list[float]] = {k: [] for k in STRATA}
    item_has: dict[str, list[bool]] = {k: [False] * n_items for k in STRATA}
    lian: list[float] = []
    bf16_ok: list[float] = []
    bf16_bad: list[float] = []
    for i in range(n_items):
        for s in seeds:
            b = bf16_seeds[s]["details"][i]
            o = other_seeds[s]["details"][i]
            bt = float(b.get("completion_tokens") or 0)
            ot = float(o.get("completion_tokens") or 0)
            bc = b.get("extractive_match", 0.0) == 1.0
            oc = o.get("extractive_match", 0.0) == 1.0
            rec = ot - bt
            if bc and oc:
                key = "both_correct"
            elif bc and not oc:
                key = "bf16_only"
            elif (not bc) and oc:
                key = "quant_only"
            else:
                key = "both_wrong"
            buckets[key].append(rec)
            item_has[key][i] = True
            if bc:
                lian.append(rec)
                bf16_ok.append(bt)
            else:
                bf16_bad.append(bt)
    out: dict[str, dict[str, float]] = {}
    for k, xs in buckets.items():
        out[k] = {
            "n": len(xs),
            "mean": (sum(xs) / len(xs) if xs else 0.0),
            "n_problems": int(sum(item_has[k])),
        }
    out["lian"] = {"n": len(lian), "mean": (sum(lian) / len(lian) if lian else 0.0)}
    out["bf16_ok"] = {"n": len(bf16_ok), "mean": (sum(bf16_ok) / len(bf16_ok) if bf16_ok else 0.0)}
    out["bf16_bad"] = {"n": len(bf16_bad), "mean": (sum(bf16_bad) / len(bf16_bad) if bf16_bad else 0.0)}
    return out


def independent_cpass_points() -> dict[tuple[str, str, str], float]:
    raw_dir = os.path.join(REPO_ROOT, "results", "measured_serving_confirmation", "raw")
    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for name in os.listdir(raw_dir):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(raw_dir, name)) as fp:
            d = json.load(fp)
        if d.get("benchmark_type") != "task_realistic_confirmation":
            continue
        key = (d["model"], d["format"], d["condition"])
        buckets[key].append(float(d["gpu_seconds_per_query"]))
    pass1 = rev.load_canonical_pass1()
    out: dict[tuple[str, str, str], float] = {}
    for (model, fmt, cond), xs in buckets.items():
        gpu = sum(xs) / len(xs)
        out[(model, fmt, cond)] = (gpu * GPU_USD_PER_SEC) / pass1[(model, fmt)]
    return out


def verify(report: dict[str, Any], serving: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    math_data = rev.load_dir(rev.MATH_DIR)
    gpqa_data = rev.load_dir(rev.GPQA_DIR)

    # Independent 2×2 means/n vs frozen token_analysis (MATH, all 6 contrasts).
    for m in rev.MODELS:
        for fmt in ["FP8", "AWQ-4", "GPTQ-4"]:
            got = independent_stratum_counts(math_data[m]["BF16"], math_data[m][fmt], rev.MATH_SEEDS, 500)
            frozen = report["math500"]["token_analysis"][f"{m}_{fmt}"]
            for k in STRATA:
                if got[k]["n"] != frozen["strata"][k]["n"]:
                    errors.append(f"{m} {fmt} {k} n {got[k]['n']} vs {frozen['strata'][k]['n']}")
                if not math.isclose(got[k]["mean"], frozen["strata"][k]["mean"], abs_tol=1e-9):
                    errors.append(f"{m} {fmt} {k} mean drift")
            if got["lian"]["n"] != frozen["lian_bf16_correct_delta"]["n"]:
                errors.append(f"{m} {fmt} lian n drift")
            if not math.isclose(got["lian"]["mean"], frozen["lian_bf16_correct_delta"]["mean"], abs_tol=1e-9):
                errors.append(f"{m} {fmt} lian mean drift")
            excess = frozen.get("mismatch_excess_vs_both_correct")
            if excess is None:
                errors.append(f"{m} {fmt} missing mismatch_excess_vs_both_correct")
            else:
                d_obs = got["bf16_only"]["mean"] - got["both_correct"]["mean"]
                if not math.isclose(d_obs, excess["mean"], abs_tol=1e-9):
                    errors.append(f"{m} {fmt} mismatch excess mean {d_obs} vs {excess['mean']}")
                if got["both_correct"]["n"] != excess["both_correct"]["n_item_seed"]:
                    errors.append(f"{m} {fmt} both_correct n_item_seed drift")
                if got["bf16_only"]["n"] != excess["bf16_only"]["n_item_seed"]:
                    errors.append(f"{m} {fmt} bf16_only n_item_seed drift")
                if got["both_correct"]["n_problems"] != excess["both_correct"]["n_problems"]:
                    errors.append(f"{m} {fmt} both_correct n_problems drift")
                if got["bf16_only"]["n_problems"] != excess["bf16_only"]["n_problems"]:
                    errors.append(f"{m} {fmt} bf16_only n_problems drift")

    # Independent GPQA Qwen AWQ clustered contrast (largest accuracy effect).
    bf16 = rev.item_correctness(gpqa_data["Qwen-7B"]["BF16"], rev.BREADTH_SEEDS, 198)
    awq = rev.item_correctness(gpqa_data["Qwen-7B"]["AWQ-4"], rev.BREADTH_SEEDS, 198)
    boot = rev.paired_delta_bootstrap(bf16, awq)
    frozen_c = next(
        c for c in report["gpqa_diamond"]["pass1_contrasts"] if c["contrast"] == "Qwen-7B BF16 vs AWQ-4"
    )
    for key in ("delta_pp", "ci95_lo_pp", "ci95_hi_pp", "ci90_lo_pp", "ci90_hi_pp", "p_value"):
        if not math.isclose(boot[key], frozen_c[key], abs_tol=1e-9):
            errors.append(f"GPQA Qwen AWQ {key}: {boot[key]} vs {frozen_c[key]}")

    # Independent hybrid Cpass point estimates.
    points = independent_cpass_points()
    for model in ["Qwen-7B", "Llama-8B"]:
        for fmt in ["BF16", "FP8", "AWQ-4", "GPTQ-4"]:
            for cond, cond_key in [
                ("A_single_stream_c1", "A_single_stream_c1"),
                ("B_batched_throughput_c8", "B_batched_throughput_c8"),
            ]:
                frozen = serving["configurations"][f"{model}_{fmt}"]["conditions"][cond_key][
                    "hybrid_scenario_cost_pass_dollars"
                ]
                got = points[(model, fmt, cond)]
                if not math.isclose(got, frozen, rel_tol=0, abs_tol=1e-9):
                    errors.append(f"Cpass {model} {fmt} {cond}: {got} vs {frozen}")

    holm18 = report.get("holm18_sensitivity")
    if holm18 is None:
        errors.append("missing holm18_sensitivity")
    else:
        pvals = []
        within = []
        for key in ("math500", "gsm8k", "gpqa_diamond"):
            for c in report[key]["pass1_contrasts"]:
                pvals.append(c["p_value"])
                within.append(bool(c["holm_significant_pass1"]))
        adj = rev.holm_adjusted_pvalues(pvals)
        holm = rev.holm(pvals)
        if holm18.get("n_contrasts") != 18:
            errors.append(f"holm18 n_contrasts {holm18.get('n_contrasts')}")
        if len(holm18.get("contrasts", [])) != 18:
            errors.append("holm18 contrasts length")
        for i, rec in enumerate(holm18.get("contrasts", [])):
            if not math.isclose(rec["holm_p_global18"], adj[i], abs_tol=1e-12):
                errors.append(f"holm18 adj p drift {rec['contrast']}")
            if rec["holm_significant_global18"] != holm[i]["significant"]:
                errors.append(f"holm18 decision drift {rec['contrast']}")
            if rec["holm_significant_within_benchmark"] != within[i]:
                errors.append(f"holm18 within-benchmark flag drift {rec['contrast']}")
            expected_change = within[i] != holm[i]["significant"]
            listed = any(ch["contrast"] == rec["contrast"] and ch["benchmark"] == rec["benchmark"] for ch in holm18.get("status_changes", []))
            if expected_change != listed:
                errors.append(f"holm18 status_change list mismatch {rec['benchmark']} {rec['contrast']}")
    return errors


def md_contrast_table(title: str, contrasts: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Contrast | Δ pp | 95% CI | p | Holm-6 | Holm-18 | 90% CI | TOST ±1 pp |",
        "|---|---:|---|---:|:---:|:---:|---|:---:|",
    ]
    for c in contrasts:
        holm = "yes" if c["holm_significant_pass1"] else "no"
        holm18 = "yes" if c.get("holm_significant_global18") else "no"
        tost = "pass" if c["tost_equiv_1pp"] else "fail"
        lines.append(
            f"| {c['contrast']} | {_fmt_pp(c['delta_pp'])} | "
            f"{_ci95(c['ci95_lo_pp'], c['ci95_hi_pp'])} | {c['p_value']:.4f} | {holm} | {holm18} | "
            f"{_ci95(c['ci90_lo_pp'], c['ci90_hi_pp'])} | {tost} |"
        )
    lines.append("")
    return lines


def md_length_table(report: dict[str, Any]) -> list[str]:
    lines = [
        "### MATH-500 length 2×2 (clustered 95% CI) and Lian BF16-correct estimand",
        "",
        "| Contrast | Both-OK Δ [95% CI] (n) | BF16-only Δ [95% CI] (n) | Quant-only Δ [95% CI] (n) | Both-wrong Δ [95% CI] (n) | Lian Δ [95% CI] (n) | BF16 len correct vs incorrect | RoM% | RoM excl near-cap% |",
        "|---|---|---|---|---|---|---|---:|---:|",
    ]
    for m in rev.MODELS:
        for fmt in ["FP8", "AWQ-4", "GPTQ-4"]:
            t = report["math500"]["token_analysis"][f"{m}_{fmt}"]
            s = t["strata"]

            def cell(key: str) -> str:
                r = s[key]
                return f"{r['mean']:+.0f} [{r['ci95_lo']:+.0f}, {r['ci95_hi']:+.0f}] (n={r['n']})"

            lian = t["lian_bf16_correct_delta"]
            ok = t["bf16_length_when_correct"]
            bad = t["bf16_length_when_incorrect"]
            lines.append(
                f"| {m} {fmt} | {cell('both_correct')} | {cell('bf16_only')} | "
                f"{cell('quant_only')} | {cell('both_wrong')} | "
                f"{lian['mean']:+.0f} [{lian['ci95_lo']:+.0f}, {lian['ci95_hi']:+.0f}] (n={lian['n']}) | "
                f"{ok['mean']:.0f} vs {bad['mean']:.0f} | "
                f"{t['ratio_of_means_pct']:+.2f} | {t['ratio_of_means_excl_nearcap_pct']:+.2f} |"
            )
    lines.append("")
    lines.append(
        "Interpretation rule: if Both-OK CI includes 0, extra length is primarily localized to mismatch pairs; "
        "if it excludes 0, 4-bit still lengthens jointly-correct traces. "
        "Mismatch excess $D$ (next table) tests whether BF16-only mean Δ exceeds Both-OK mean Δ on the same clustered resamples. "
        "Do not call a BF16-failure length gap a Liu–Lian reconciliation. $D$ is not causal."
    )
    lines.append("")
    return lines


def md_mismatch_excess(report: dict[str, Any]) -> list[str]:
    lines = [
        "### MATH-500 mismatch excess $D$ (clustered 95% CI)",
        "",
        "$D$ = mean(token Δ | BF16-only) − mean(token Δ | Both-OK). Same item-resamples; empty-stratum replicates skipped. Not causal.",
        "",
        "| Contrast | Both-OK Δ [95% CI] | BF16-only Δ [95% CI] | **Mismatch excess $D$ [95% CI]** | p | n Both-OK (pairs/problems) | n BF16-only (pairs/problems) | boot valid/skipped |",
        "|---|---|---|---|---:|---|---|---|",
    ]
    for m in rev.MODELS:
        for fmt in ["FP8", "AWQ-4", "GPTQ-4"]:
            t = report["math500"]["token_analysis"][f"{m}_{fmt}"]
            s = t["strata"]
            ex = t["mismatch_excess_vs_both_correct"]

            def cell(key: str) -> str:
                r = s[key]
                return f"{r['mean']:+.0f} [{r['ci95_lo']:+.0f}, {r['ci95_hi']:+.0f}]"

            lines.append(
                f"| {m} {fmt} | {cell('both_correct')} | {cell('bf16_only')} | "
                f"**{ex['mean']:+.0f} [{ex['ci95_lo']:+.0f}, {ex['ci95_hi']:+.0f}]** | "
                f"{ex['p_value']:.4f} | "
                f"{ex['both_correct']['n_item_seed']}/{ex['both_correct']['n_problems']} | "
                f"{ex['bf16_only']['n_item_seed']}/{ex['bf16_only']['n_problems']} | "
                f"{ex['n_boot_valid']}/{ex['n_boot_skipped_empty_stratum']} |"
            )
    lines.append("")
    lines.append(
        "If the $D$ CI is entirely >0, mismatch-associated lengthening is larger than jointly-correct lengthening. "
        "If it includes 0, there is no clear difference between those conditional stratum means."
    )
    lines.append("")
    return lines


def md_holm18(report: dict[str, Any]) -> list[str]:
    holm18 = report["holm18_sensitivity"]
    lines = [
        "### Holm-18 sensitivity (secondary; primary remains Holm-6 within each benchmark)",
        "",
        holm18["note"],
        "",
        "| Benchmark | Contrast | raw p | Holm-6 | Holm-18 adj p | Holm-18 | Status change |",
        "|---|---|---:|:---:|---:|:---:|---|",
    ]
    for rec in holm18["contrasts"]:
        ch = rec.get("status_change", "none")
        lines.append(
            f"| {rec['benchmark']} | {rec['contrast']} | {rec['p_value']:.4f} | "
            f"{'yes' if rec['holm_significant_within_benchmark'] else 'no'} | "
            f"{rec['holm_p_global18']:.4f} | "
            f"{'yes' if rec['holm_significant_global18'] else 'no'} | {ch} |"
        )
    lines.append("")
    if holm18["status_changes"]:
        lines.append("Contrasts whose significance status changes:")
        lines.append("")
        for ch in holm18["status_changes"]:
            lines.append(
                f"- **{ch['benchmark']} {ch['contrast']}**: p={ch['p_value']:.4f}; {ch['status_change']}"
            )
        lines.append("")
    else:
        lines.append("No within-benchmark vs global-18 significance status changes.")
        lines.append("")
    return lines


def md_serving(serving: dict[str, Any]) -> list[str]:
    lines = [
        "### Hybrid scenario Cost-of-Pass (Condition A and B)",
        "",
        "| Cell | A tok/s | A GPU-s/q | A $C_{pass}$ [95% CI] | B tok/s | B GPU-s/q | B $C_{pass}$ [95% CI] | B Δ% vs BF16 [95% CI] |",
        "|---|---:|---:|---|---:|---:|---|---|",
    ]
    for model in ["Qwen-7B", "Llama-8B"]:
        for fmt in ["BF16", "FP8", "AWQ-4", "GPTQ-4"]:
            cfg = serving["configurations"][f"{model}_{fmt}"]
            a = cfg["conditions"]["A_single_stream_c1"]
            b = cfg["conditions"]["B_batched_throughput_c8"]
            a_ci = a["hybrid_cpass_ci95"]
            b_ci = b["hybrid_cpass_ci95"]
            if fmt == "BF16":
                delta = "anchor"
            else:
                dci = b["hybrid_cpass_delta_vs_bf16_ci95_pct"]
                delta = f"{b['cost_pass_delta_vs_bf16_pct']:+.1f}% [{dci[0]:+.1f}, {dci[1]:+.1f}]"
            lines.append(
                f"| {model} {fmt} | {a['mean_tokens_per_second']:.2f} | {a['mean_gpu_seconds_per_query']:.2f} | "
                f"${a['hybrid_scenario_cost_pass_dollars']:.4f} [{a_ci[0]:.4f}, {a_ci[1]:.4f}] | "
                f"{b['mean_tokens_per_second']:.2f} | {b['mean_gpu_seconds_per_query']:.2f} | "
                f"${b['hybrid_scenario_cost_pass_dollars']:.4f} [{b_ci[0]:.4f}, {b_ci[1]:.4f}] | {delta} |"
            )
    lines.append("")
    lines.append("Rank order (1 = lowest aggregate cost proxy):")
    lines.append("")
    for model in ["Qwen-7B", "Llama-8B"]:
        block = serving["ranking_tables"][model]
        lines.append(f"- **{model}:** {block['headline']}")
        lines.append(f"  - token-proxy $C_{{pass}}$: {' > '.join(block['rank_order']['proxy_cpass_65toks'])}")
        lines.append(f"  - Condition A $C_{{pass}}$: {' > '.join(block['rank_order']['hybrid_cpass_A'])}")
        lines.append(f"  - Condition B $C_{{pass}}$: {' > '.join(block['rank_order']['hybrid_cpass_B'])}")
    lines.append("")
    fp8 = serving["qwen_fp8_condition_b"]
    lines.append("### Qwen-7B FP8 Condition B (all five repeats)")
    lines.append("")
    lines.append(
        f"Mean {fp8['mean_tokens_per_second']:.2f} ± {fp8['std_tokens_per_second']:.2f} tok/s; "
        f"median {fp8['median_tokens_per_second']:.2f}; "
        f"IQR [{fp8['iqr_tokens_per_second'][0]:.2f}, {fp8['iqr_tokens_per_second'][1]:.2f}]."
    )
    lines.append("")
    lines.append("| Rep | tok/s | GPU-s/q | hybrid $C_{pass}$ | regime |")
    lines.append("|---|---:|---:|---:|---|")
    for row in fp8["repetitions"]:
        lines.append(
            f"| {row['repetition']} | {row['tokens_per_second']:.2f} | "
            f"{row['gpu_seconds_per_query']:.4f} | ${row['hybrid_cpass_dollars']:.4f} | {row['regime']} |"
        )
    slow, mid, fast = fp8["slow_regime"], fp8.get("mid_regime", {}), fp8["fast_regime"]
    lines.append("")
    lines.append(
        f"Slow regime n={slow['n']} ~{slow['mean_tokens_per_second']:.0f} tok/s, "
        f"$C_{{pass}}$ ${slow['hybrid_cpass_dollars']:.4f}. "
        f"Mid regime n={mid.get('n', 0)} ~{mid.get('mean_tokens_per_second', 0):.0f} tok/s, "
        f"$C_{{pass}}$ ${mid.get('hybrid_cpass_dollars', 0):.4f}. "
        f"Fast regime n={fast['n']} ~{fast['mean_tokens_per_second']:.0f} tok/s, "
        f"$C_{{pass}}$ ${fast['hybrid_cpass_dollars']:.4f}."
    )
    llama_b = serving["configurations"]["Llama-8B_GPTQ-4"]["conditions"]["B_batched_throughput_c8"]
    llama_bf = serving["configurations"]["Llama-8B_BF16"]["conditions"]["B_batched_throughput_c8"]
    rel = (llama_b["mean_tokens_per_second"] / llama_bf["mean_tokens_per_second"] - 1.0) * 100.0
    lines.append("")
    lines.append(
        f"Llama GPTQ-4 Condition B mean throughput is {rel:+.2f}% vs Llama BF16 "
        f"({llama_b['mean_tokens_per_second']:.2f} vs {llama_bf['mean_tokens_per_second']:.2f} tok/s). "
        "Do not say statistically tied."
    )
    lines.append("")
    return lines


def md_modal(modal_report: dict[str, Any]) -> list[str]:
    lines = [
        "### Modal selective risk with Wilson / Clopper–Pearson intervals",
        "",
        "| Cell | τ | served | errors | risk% | bootstrap 95% | Wilson 95% | CP 95% | 0 observed ≠ 0 true |",
        "|---|---|---:|---:|---:|---|---|---|:---:|",
    ]
    for model, fmt in modal.CONFIG_NAMES:
        cfg = modal_report["configurations"][f"{model}_{fmt}"]
        for tau in (3, 4, 5):
            th = cfg["thresholds"][f"tau_{tau}"]
            err = th["selective_risk_errors"]
            risk = th["selective_risk"] * 100.0
            b = th["selective_risk_ci_95"]
            w = th["selective_risk_wilson_ci_95"]
            cp = th["selective_risk_clopper_pearson_ci_95"]
            flag = "yes" if th.get("zero_observed_not_zero_true") else "no"
            lines.append(
                f"| {model} {fmt} | {th['threshold']} | {th['served_count']} | {err} | {risk:.2f} | "
                f"[{b[0]*100:.2f}, {b[1]*100:.2f}] | [{w[0]*100:.2f}, {w[1]*100:.2f}] | "
                f"[{cp[0]*100:.2f}, {cp[1]*100:.2f}] | {flag} |"
            )
    lines.append("")
    lines.append("Zero observed errors does not imply zero true selective risk.")
    lines.append("")
    return lines


def render_major_revision_markdown(
    report: dict[str, Any], serving: dict[str, Any], modal_report: dict[str, Any]
) -> str:
    lines = [
        "# Major revision tables (frozen analysis)",
        "",
        "Generated by `scripts/analysis/emit_major_revision_tables.py` after independent recompute from raw JSON.",
        "Do not edit numbers by hand. Rewrite LaTeX from this file.",
        "",
        "## Pass@1 clustered contrasts",
        "",
    ]
    lines.extend(md_contrast_table("MATH-500 (add 90% TOST CI to Table 2)", report["math500"]["pass1_contrasts"]))
    lines.extend(md_contrast_table("GSM8K", report["gsm8k"]["pass1_contrasts"]))
    lines.extend(md_contrast_table("GPQA Diamond", report["gpqa_diamond"]["pass1_contrasts"]))
    lines.extend(md_holm18(report))
    lines.extend(md_length_table(report))
    lines.extend(md_mismatch_excess(report))
    lines.extend(md_serving(serving))
    lines.extend(md_modal(modal_report))
    lines.extend(
        [
            "## Wording bans for the rewrite",
            "",
            "- architecture-dependent, format-induced failures, reconcile, statistically tied,",
            "  cheapest as a law, near-cap terminations, true Pareto, 18.7, unqualified −45.9% headline",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def render_validation_markdown(report: dict[str, Any]) -> str:
    changes = report["holm18_sensitivity"]["status_changes"]
    valid = [
        "# Major revision validation",
        "",
        "CPU-only independent recompute via `scripts/analysis/emit_major_revision_tables.py`.",
        "Does not replace `revision_reanalysis.py --check`.",
        "",
        "- Independent 2×2 n/mean, Lian mean, and mismatch-excess point estimate: **OK**",
        "- Independent GPQA Qwen AWQ clustered contrast: **OK**",
        "- Independent hybrid $C_{pass}$ points: **OK**",
        "- Holm-18 adjusted p / decisions vs recomputed Holm: **OK**",
        "",
        "## Holm-18 status changes",
        "",
    ]
    if changes:
        valid.extend(
            f"- {ch['benchmark']} {ch['contrast']}: {ch['status_change']} "
            f"(p={ch['p_value']:.4f}, holm18_p={ch['holm_p_global18']:.4f})"
            for ch in changes
        )
    else:
        valid.append("- none")
    valid.append("")
    return "\n".join(valid) + "\n"


def _read_text(path: str) -> str:
    with open(path) as fp:
        return fp.read()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Independently recompute major-revision tables from raw JSON."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare generated markdown with frozen tables; fail on drift. Does not write.",
    )
    args = parser.parse_args(argv)

    with open(REPORT) as fp:
        report = json.load(fp)
    with open(SERVING) as fp:
        serving = json.load(fp)
    with open(MODAL) as fp:
        modal_report = json.load(fp)
    modal.attach_binomial_risk_intervals_report(modal_report)

    errs = verify(report, serving)
    if errs:
        print("INDEPENDENT VERIFY FAILED:", file=sys.stderr)
        for e in errs:
            print(f"  {e}", file=sys.stderr)
        return 1

    tables = render_major_revision_markdown(report, serving, modal_report)
    valid = render_validation_markdown(report)

    if args.check:
        missing = [p for p in (OUT_MD, OUT_VALID) if not os.path.isfile(p)]
        if missing:
            print("ERROR: missing frozen table file(s): " + ", ".join(missing), file=sys.stderr)
            return 1
        diffs: list[str] = []
        if tables != _read_text(OUT_MD):
            diffs.append(f"drift vs {OUT_MD}")
        if valid != _read_text(OUT_VALID):
            diffs.append(f"drift vs {OUT_VALID}")
        if diffs:
            print("ERROR: generated tables do not match frozen files:", file=sys.stderr)
            for d in diffs:
                print(f"  {d}", file=sys.stderr)
            return 1
        print(f"OK: generated tables match {OUT_MD} and {OUT_VALID}")
        return 0

    with open(OUT_MD, "w") as fp:
        fp.write(tables)
    with open(OUT_VALID, "w") as fp:
        fp.write(valid)
    print(f"OK: independent recompute matches frozen JSON. Wrote {OUT_MD}")
    print(f"Wrote {OUT_VALID}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
