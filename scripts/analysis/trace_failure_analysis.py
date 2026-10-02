"""CPU-only failure-mode and length-signal analysis of the MATH-500 campaign (no GPU, no frozen result changed).

  1. Failure-mode taxonomy. Every incorrect completion is assigned exactly one class, in priority order:
       truncated  completion within 16 tokens of its per-item output cap (32,768 - prompt tokens)
       loop       identical-word run >= 20 (campaign validator flag)
       no_box     finished, no \\boxed marker
       wrong      finished, boxed, wrong answer
     For each quantized checkpoint the error-rate difference to BF16 (percentage points of all completions) is
     split by class, with item-level 95% CIs. Row alignment of compact JSON and raw rows is asserted.
  2. Paired transition table: BF16 outcome class -> quantized outcome class for same-seed same-item pairs.
     Same-seed pairing is an arbitrary coupling (see the seed placebo), so the marginal rate differences of (1)
     are the inferential quantity; the table is descriptive.
  3. Does length predict failure? Pooled AUROC of "shorter completion is correct" and an item-stratified AUROC
     (only correct-vs-incorrect pairs within the same item, which removes item difficulty), per cell and
     quantized-minus-BF16, item-level 95% bootstrap CIs.

Writes results/reports/trace_failure_analysis.json.  Usage: python scripts/analysis/trace_failure_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FAMILIES = {"Qwen-7B": "DeepSeek-R1-Distill-Qwen-7B", "Llama-8B": "DeepSeek-R1-Distill-Llama-8B"}
FORMATS = ["BF16", "FP8", "AWQ-4", "GPTQ-4"]
SEEDS = [42, 43, 44, 45, 46]
CLASSES = ["correct", "truncated", "loop", "no_box", "wrong"]
MAX_LEN, TOL, B, RNG_SEED = 32768, 16, 10_000, 0


def cell(f, m):
    return FAMILIES[f] if m == "BF16" else f"{FAMILIES[f]}-{m}"


def compact(f, m, s):
    return json.loads((REPO / "results/math500" / f"{cell(f, m)}_math500_n500_seed{s}.json").read_text())["details"]


def raw(f, m, s):
    return json.loads((REPO / "outputs-hpc-campaign-2026-08-14/inference" / f"{cell(f, m)}-seed{s}" / "MATH-500.jsonl").read_text())


def classify(d, cap):
    ok = np.array([x["extractive_match"] == 1.0 for x in d])
    tok = np.array([x["completion_tokens"] for x in d])
    loop = np.array([bool(x["repetition_flag"]) for x in d])
    boxed = np.array([bool(x["boxed"]) for x in d])
    cls = np.full(len(d), 4)  # wrong
    cls[~boxed] = 3
    cls[loop] = 2
    cls[tok >= cap - TOL] = 1
    cls[ok] = 0
    return cls, tok


def ci(v):
    return [round(float(x), 2) for x in np.percentile(v, [2.5, 97.5])]


def boot_rows(X, rng, chunk=500):
    """X: items x k matrix of per-item values; returns B bootstrap means (B x k)."""
    n = X.shape[0]
    out = []
    for _ in range(B // chunk):
        idx = rng.integers(0, n, (chunk, n))
        out.append(X[idx].mean(1))
    return np.concatenate(out)


def main() -> int:
    from transformers import AutoTokenizer
    rng = np.random.default_rng(RNG_SEED)
    rep = {"inputs": {"B": B, "rng_seed": RNG_SEED, "cap_tolerance_tokens": TOL, "classes": CLASSES}}
    tax, trans, auc = {}, {}, {}
    for fam in FAMILIES:
        tok = AutoTokenizer.from_pretrained(REPO / "models" / FAMILIES[fam])
        C, T = {}, {}
        for m in FORMATS:
            cls_s, tok_s = [], []
            for s in SEEDS:
                d, r = compact(fam, m, s), raw(fam, m, s)
                assert len(d) == len(r) == 500
                if s == SEEDS[0]:
                    cap = MAX_LEN - np.array([len(tok(x["full_prompt"], add_special_tokens=False)["input_ids"]) for x in r])
                c, t = classify(d, cap)
                cls_s.append(c); tok_s.append(t)
            C[m], T[m] = np.array(cls_s), np.array(tok_s)  # seeds x items
        # 1. taxonomy: per-item error-rate by class (items x classes), seeds averaged
        rate = {m: np.stack([(C[m] == k).mean(0) for k in range(5)], 1) for m in FORMATS}
        for m in FORMATS:
            tax[f"{fam}|{m}"] = {c: round(float(rate[m][:, k].mean()) * 100, 2) for k, c in enumerate(CLASSES)}
        for m in FORMATS[1:]:
            D = rate[m] - rate["BF16"]
            bs = boot_rows(D, rng) * 100
            tax[f"{fam}|{m}|minus_BF16_pp"] = {c: [round(float(D[:, k].mean()) * 100, 2), ci(bs[:, k])] for k, c in enumerate(CLASSES) if c != "correct"}
            tax[f"{fam}|{m}|minus_BF16_pp"]["all_errors"] = [round(float(-D[:, 0].mean()) * 100, 2), ci(-bs[:, 0])]
            # 2. paired transitions (same seed, same item)
            M = np.zeros((5, 5), int)
            for s in range(len(SEEDS)):
                np.add.at(M, (C["BF16"][s], C[m][s]), 1)
            trans[f"{fam}|{m}"] = {"rows": "BF16 class", "cols": "quantized class", "classes": CLASSES, "counts": M.tolist()}
        # 3. length as a failure predictor
        from scipy.stats import rankdata
        WP = {}
        for m in FORMATS:
            ok = (C[m] == 0)
            L = T[m].astype(float)
            # item-stratified: wins over correct-vs-incorrect pairs within each item, across the 5 seeds
            wins = np.zeros(500); pairs = np.zeros(500)
            for i in range(500):
                lc, lw = L[ok[:, i], i], L[~ok[:, i], i]
                if len(lc) and len(lw):
                    g = (lw[None, :] > lc[:, None]).sum() + 0.5 * (lw[None, :] == lc[:, None]).sum()
                    wins[i], pairs[i] = g, len(lc) * len(lw)
            # pooled AUROC per bootstrap draw
            def pooled(idx):
                l, o = L[:, idx].ravel(), ok[:, idx].ravel()
                if o.all() or (~o).all():
                    return np.nan
                r = rankdata(-l)  # high rank = short
                n1, n0 = o.sum(), (~o).sum()
                return (r[o].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
            pb = np.array([pooled(rng.integers(0, 500, 500)) for _ in range(2000)])
            WP[m] = (wins, pairs)
            auc[f"{fam}|{m}"] = {"pooled_auroc": [round(float(pooled(np.arange(500))), 3), ci(pb)],
                                "items_with_both_outcomes": int((pairs > 0).sum())}
        # item-stratified AUROC with shared item resamples so quantized-minus-BF16 is a paired contrast
        draws = {m: [] for m in FORMATS}
        diffs = {m: [] for m in FORMATS[1:]}
        for _ in range(B // 500):
            idx = rng.integers(0, 500, (500, 500))
            a = {m: WP[m][0][idx].sum(1) / WP[m][1][idx].sum(1) for m in FORMATS}
            for m in FORMATS:
                draws[m].append(a[m])
            for m in FORMATS[1:]:
                diffs[m].append(a[m] - a["BF16"])
        for m in FORMATS:
            w, p_ = WP[m]
            auc[f"{fam}|{m}"]["item_stratified_auroc"] = [round(float(w.sum() / p_.sum()), 3), ci(np.concatenate(draws[m]))]
        for m in FORMATS[1:]:
            w, p_ = WP[m]; wb, pb_ = WP["BF16"]
            auc[f"{fam}|{m}"]["item_stratified_auroc_minus_BF16"] = [round(float(w.sum() / p_.sum() - wb.sum() / pb_.sum()), 3), ci(np.concatenate(diffs[m]))]
    rep["failure_taxonomy_pct_of_completions"] = tax
    rep["paired_class_transitions"] = trans
    rep["length_as_failure_predictor"] = auc
    out = REPO / "results/reports/trace_failure_analysis.json"
    out.write_text(json.dumps(rep, indent=1) + "\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
