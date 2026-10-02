"""CPU-only analyses added after the second independent review (no GPU, no frozen result changed).

  1. Adaptive agreement sampling on MATH-500. Samples are drawn one at a time (all 5! = 120 orderings of the five
     stored seeds are averaged) and the equivalence relation is the campaign one used by revision_sensitivities.py.
       fixed2       serve iff the first 2 samples agree (the k=2 unanimity rule)
       pair@N       serve the first answer that two of the first N samples share; abstain if no pair by N (N = 3, 5)
       2-then-3of5  draw 2; serve if they agree; otherwise draw 3 more and serve the unique mode only if it has >= 3 of 5
     Cost is the realised sum of completion tokens of the samples actually drawn. Item-level bootstrap CIs.
  2. Reflection markers in the stored completions (wait / hmm / alternatively / check-verify), per 1,000 tokens and
     per completion, by cell, split by truncated vs finished completions; quantized-minus-BF16 item-level CIs.
  3. Graded equivalence: posterior probability that a pass@1 contrast lies within +-1 pp (and +-2 pp) under a flat
     Dirichlet Bayesian bootstrap over the paired item deltas. Conditional on the archived seeds; not a hypothesis test.

Writes results/reports/adaptive_reflection_rope.json.  Usage: python scripts/analysis/adaptive_reflection_rope.py
"""
from __future__ import annotations

import functools
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RECOVERED = REPO / "results/recovered/math500_modal_inputs.jsonl"
FAMILIES = {"Qwen-7B": "DeepSeek-R1-Distill-Qwen-7B", "Llama-8B": "DeepSeek-R1-Distill-Llama-8B"}
FORMATS = ["BF16", "FP8", "AWQ-4", "GPTQ-4"]
SEEDS = [42, 43, 44, 45, 46]
BENCH = {"math500": (500, SEEDS), "gsm8k": (1319, SEEDS[:3]), "gpqa": (198, SEEDS[:3])}
MAX_LEN, TOL = 32768, 16
B, RNG_SEED = 10_000, 0


def cell(f, m):
    return FAMILIES[f] if m == "BF16" else f"{FAMILIES[f]}-{m}"


def ci(v, nd=2):
    return [round(float(x), nd) for x in np.percentile(v, [2.5, 97.5])]


# ----------------------------------------------------------------------------------------------- 1. adaptive
def equivalence():
    import sympy
    from math_verify import verify

    @functools.lru_cache(maxsize=None)
    def parse(t):
        try:
            return sympy.sympify(t, evaluate=True)
        except Exception:
            return None

    @functools.lru_cache(maxsize=None)
    def eq(x, y):
        if x == y:
            return True
        px, py = parse(x), parse(y)
        if px is None or py is None:
            return False
        try:
            return bool(verify(px, py) and verify(py, px))
        except Exception:
            return False
    return eq


def adaptive(rng):
    eq = equivalence()
    recs = defaultdict(dict)
    for line in RECOVERED.open():
        r = json.loads(line)
        key = r["extracted_pred_repr"][0] if r["extracted_pred_repr"] else None
        recs[(r["model"], r["format"])][(r["problem_index"], r["seed"])] = (key, r["campaign_extractive_match"], r["completion_tokens"])
    perms = list(itertools.permutations(range(5)))
    out = {}
    for fam in FAMILIES:
        for fmt in FORMATS:
            cell_ = recs[(fam, fmt)]
            E = np.zeros((500, 5, 5), bool); C = np.zeros((500, 5)); T = np.zeros((500, 5))
            for i in range(500):
                for a, sa in enumerate(SEEDS):
                    ka, ca, ta = cell_[(i, sa)]
                    C[i, a], T[i, a] = ca, ta
                    for b, sb in enumerate(SEEDS):
                        kb = cell_[(i, sb)][0]
                        E[i, a, b] = ka is not None and kb is not None and eq(ka, kb)
            policies = ["fixed2", "pair@3", "pair@5", "2-then-3of5"]
            served = {p: np.zeros(500) for p in policies}
            wrong = {p: np.zeros(500) for p in policies}
            toks = {p: np.zeros(500) for p in policies}
            for i in range(500):
                for perm in perms:
                    o = perm
                    # fixed2 and the pair@N rules
                    first_pair = None
                    for n in range(2, 6):
                        for a in range(n - 1):
                            if E[i, o[a], o[n - 1]]:
                                first_pair = (n, o[a]); break
                        if first_pair:
                            break
                    # fixed2
                    if E[i, o[0], o[1]]:
                        served["fixed2"][i] += 1; wrong["fixed2"][i] += C[i, o[0]] == 0
                    toks["fixed2"][i] += T[i, o[0]] + T[i, o[1]]
                    for N, name in ((3, "pair@3"), (5, "pair@5")):
                        if first_pair and first_pair[0] <= N:
                            n, a = first_pair
                            served[name][i] += 1; wrong[name][i] += C[i, a] == 0
                            toks[name][i] += sum(T[i, o[j]] for j in range(n))
                        else:
                            toks[name][i] += sum(T[i, o[j]] for j in range(N))
                    # 2-then-3of5
                    if E[i, o[0], o[1]]:
                        served["2-then-3of5"][i] += 1; wrong["2-then-3of5"][i] += C[i, o[0]] == 0
                        toks["2-then-3of5"][i] += T[i, o[0]] + T[i, o[1]]
                    else:
                        toks["2-then-3of5"][i] += T[i].sum()
                        # unique mode with >= 3 of the 5 samples
                        sizes = [(int(E[i, a].sum()), a) for a in range(5)]
                        best = max(s for s, _ in sizes)
                        tops = {frozenset(np.flatnonzero(E[i, a]).tolist()) for s, a in sizes if s == best}
                        if best >= 3 and len(tops) == 1:
                            a = [a for s, a in sizes if s == best][0]
                            served["2-then-3of5"][i] += 1; wrong["2-then-3of5"][i] += C[i, a] == 0
            res = {}
            n = len(perms)
            for p in policies:
                s_, w_, t_ = served[p] / n, wrong[p] / n, toks[p] / n
                idx = rng.integers(0, 500, (2000, 500))
                cov = s_[idx].mean(1); risk = w_[idx].sum(1) / np.maximum(s_[idx].sum(1), 1e-9); tk = t_[idx].mean(1)
                res[p] = {"coverage_pct": [round(float(s_.mean()) * 100, 2), ci(cov * 100)],
                          "risk_pct": [round(float(w_.sum() / s_.sum()) * 100, 2), ci(risk * 100)],
                          "tokens_per_item": [round(float(t_.mean()), 0), ci(tk, 0)]}
            out[f"{fam}|{fmt}"] = res
    return out


# ----------------------------------------------------------------------------------------------- 2. reflection
MARK = {
    "wait": re.compile(r"\bwait\b", re.I),
    "hmm": re.compile(r"\bhmm+\b", re.I),
    "alternatively": re.compile(r"\balternatively\b", re.I),
    "check": re.compile(r"\b(?:double-check|let me (?:check|verify|recheck)|verify)\b", re.I),
}


def reflection(rng):
    from transformers import AutoTokenizer
    out = {}
    for fam in FAMILIES:
        tok = AutoTokenizer.from_pretrained(REPO / "models" / FAMILIES[fam])
        per = {}
        for fmt in FORMATS:
            cnt = {k: np.zeros((5, 500)) for k in MARK}
            ntok = np.zeros((5, 500)); trunc = np.zeros((5, 500), bool); ok = np.zeros((5, 500), bool)
            for si, s in enumerate(SEEDS):
                raw = json.loads((REPO / "outputs-hpc-campaign-2026-08-14/inference" / f"{cell(fam, fmt)}-seed{s}" / "MATH-500.jsonl").read_text())
                det = json.loads((REPO / "results/math500" / f"{cell(fam, fmt)}_math500_n500_seed{s}.json").read_text())["details"]
                if si == 0:
                    cap = MAX_LEN - np.array([len(tok(r["full_prompt"], add_special_tokens=False)["input_ids"]) for r in raw])
                for i, (r, d) in enumerate(zip(raw, det)):
                    for k, rx in MARK.items():
                        cnt[k][si, i] = len(rx.findall(r["generated_text"]))
                    ntok[si, i] = d["completion_tokens"]; trunc[si, i] = d["completion_tokens"] >= cap[i] - TOL
                    ok[si, i] = d["extractive_match"] == 1.0
            per[fmt] = (cnt, ntok, trunc, ok)
        res = {}
        for fmt in FORMATS:
            cnt, ntok, trunc, ok = per[fmt]
            tot = sum(cnt.values())
            fin = ~trunc
            res[fmt] = {
                "wait_per_1k_tokens": round(float(cnt["wait"].sum() / ntok.sum() * 1000), 3),
                "all_markers_per_1k_tokens": round(float(tot.sum() / ntok.sum() * 1000), 3),
                "wait_per_completion": round(float(cnt["wait"].mean()), 2),
                "markers_per_1k_finished": round(float(tot[fin].sum() / ntok[fin].sum() * 1000), 3),
                "markers_per_1k_truncated": round(float(tot[trunc].sum() / ntok[trunc].sum() * 1000), 3) if trunc.any() else None,
                "n_truncated": int(trunc.sum()),
                "markers_per_1k_correct": round(float(tot[ok].sum() / ntok[ok].sum() * 1000), 3),
                "markers_per_1k_incorrect_finished": round(float(tot[~ok & fin].sum() / ntok[~ok & fin].sum() * 1000), 3),
            }
        base_c = sum(per["BF16"][0].values()).mean(0); base_n = per["BF16"][1].mean(0)
        for fmt in FORMATS[1:]:
            cq = sum(per[fmt][0].values()).mean(0); nq = per[fmt][1].mean(0)
            # ratio-of-sums of per-item seed means, item bootstrap of the density difference (per 1k tokens)
            idx = rng.integers(0, 500, (B, 500))
            dens_q = cq[idx].sum(1) / nq[idx].sum(1) * 1000
            dens_b = base_c[idx].sum(1) / base_n[idx].sum(1) * 1000
            dd = dens_q - dens_b
            dc = cq - base_c
            bc = dc[idx].mean(1)
            res[fmt]["minus_BF16"] = {
                "markers_per_1k_tokens": [round(float(cq.sum() / nq.sum() * 1000 - base_c.sum() / base_n.sum() * 1000), 3), ci(dd, 3)],
                "markers_per_completion": [round(float(dc.mean()), 2), ci(bc, 2)]}
        out[fam] = res
    return out


# ----------------------------------------------------------------------------------------------- 3. ROPE
def rope(rng):
    out = {}
    for bench, (n, seeds) in BENCH.items():
        tag = {"gpqa": "gpqadiamond"}.get(bench, bench)
        for fam in FAMILIES:
            def acc(m):
                return np.array([[float(x["extractive_match"]) for x in json.loads((REPO / "results" / bench / f"{cell(fam, m)}_{tag}_n{n}_seed{s}.json").read_text())["details"]] for s in seeds]).mean(0)
            base = acc("BF16")
            for m in FORMATS[1:]:
                d = (acc(m) - base) * 100
                w = rng.dirichlet(np.ones(n), size=B)
                post = w @ d
                out[f"{bench}|{fam}|{m}"] = {
                    "delta_pp": round(float(d.mean()), 2),
                    "p_within_1pp": round(float((np.abs(post) < 1).mean()), 3),
                    "p_within_2pp": round(float((np.abs(post) < 2).mean()), 3),
                    "p_drop_gt_1pp": round(float((post < -1).mean()), 3)}
    return out


def main():
    rng = np.random.default_rng(RNG_SEED)
    rep = {"inputs": {"B": B, "rng_seed": RNG_SEED, "orderings": 120,
                      "note": "exploratory; conditional on the archived generation seeds"}}
    rep["rope_flat_dirichlet"] = rope(rng); print("rope done", flush=True)
    rep["reflection_markers"] = reflection(rng); print("reflection done", flush=True)
    rep["adaptive_agreement"] = adaptive(rng); print("adaptive done", flush=True)
    out = REPO / "results/reports/adaptive_reflection_rope.json"
    out.write_text(json.dumps(rep, indent=1) + "\n"); print("wrote", out)


if __name__ == "__main__":
    main()
