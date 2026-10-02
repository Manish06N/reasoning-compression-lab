"""CPU-only checks added after the independent audit of 2 Oct 2026 (no GPU, no frozen result changed).

Run under the FROZEN runtime environment (qrm-official: transformers 4.47.x, tokenizers 0.21.x) because prompt and
completion token counts depend on the tokenizer implementation. Newer releases (transformers 5.x, LlamaTokenizer) tokenize
Llama text differently and give wrong prompt lengths (28-747 instead of 27-776) and completion counts (about 220 tokens off).

  1. Tokenizers: prompt-length ranges, `<think>\\n` marker ids, AWQ prompt + suffix == BF16 prompt, and (with --full) exact
     recount of every stored completion_tokens from the stored text (56,408 completions).
  2. Exact cap: completion_tokens == 32,768 - prompt_tokens (vLLM stops at max_model_len), per benchmark; cap band within 16.
  3. Serving aggregation: campaign-length seconds and item-bootstrap rank-1 frequencies with tok/s as the mean of per-repeat
     ratios (as in the paper) versus total tokens / total time.
  4. Paired item-level intervals for the BF16-correct conditional minus the seed placebo, and for the change in the Llama
     AWQ-4 gap under maj@5 (shared item resamples; stdlib Random(0) index stream, B = 10,000).
  5. Llama FP8 Condition A: measured subset length versus 20,000 resampled draws of the campaign completions.

Writes results/reports/audit_checks.json.   python scripts/analysis/audit_checks.py [--full]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np

REPO = Path(__file__).resolve().parents[2]
FAM = {"Qwen": "DeepSeek-R1-Distill-Qwen-7B", "Llama": "DeepSeek-R1-Distill-Llama-8B"}
FMT = ["BF16", "FP8", "AWQ-4", "GPTQ-4"]
BENCH = {"math500": ("outputs-hpc-campaign-2026-08-14", "MATH-500.jsonl", "math500_n500", range(42, 47)),
         "gsm8k": ("outputs-hpc-breadth-gsm8k-2026-08-15", "GSM8K.jsonl", "gsm8k_n1319", range(42, 45)),
         "gpqa": ("outputs-hpc-breadth-gpqa-2026-08-16", "GPQA-Diamond.jsonl", "gpqadiamond_n198", range(42, 45))}
MAX_LEN, TOL, B = 32768, 16, 10_000


def cell(f, m):
    return FAM[f] if m == "BF16" else f"{FAM[f]}-{m}"


def details(bench, f, m, s):
    d, _, tag, _ = BENCH[bench]
    res = {"math500": "math500", "gsm8k": "gsm8k", "gpqa": "gpqa"}[bench]
    return json.loads((REPO / "results" / res / f"{cell(f, m)}_{tag}_seed{s}.json").read_text())["details"]


def raw(bench, f, m, s):
    d, fn, _, _ = BENCH[bench]
    return json.loads((REPO / d / "inference" / f"{cell(f, m)}-seed{s}" / fn).read_text())


def acc(f, m):
    return np.array([[float(x["extractive_match"] == 1.0) for x in details("math500", f, m, s)] for s in range(42, 47)])


def tok(f, m):
    return np.array([[float(x["completion_tokens"]) for x in details("math500", f, m, s)] for s in range(42, 47)])


def weights(n, seed=0):
    rng = random.Random(seed)
    W = np.zeros((B, n))
    for b in range(B):
        for _ in range(n):
            W[b, rng.randrange(n)] += 1
    return W


def ci95(v):
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


# ------------------------------------------------------------------------------------------------ 1 + 2 tokenizers
def tokenizer_checks(full: bool) -> dict:
    import transformers
    from transformers import AutoTokenizer
    out = {"transformers": transformers.__version__, "prompt_ranges": {}, "marker_ids": {}, "awq_suffix_equal": {}, "exact_cap": {}, "cap_band_math500": {}}
    if not transformers.__version__.startswith("4.47."):
        out["warning"] = "NOT the frozen runtime: token counts are not reliable"
    for fam in FAM:
        tk = AutoTokenizer.from_pretrained(REPO / "models" / FAM[fam])
        out["marker_ids"][fam] = tk.encode("<think>\n", add_special_tokens=False)
        plen_all = {}
        for fmt in FMT:
            c = cell(fam, fmt)
            r42 = raw("math500", fam, fmt, 42)
            pl = [len(tk(r["full_prompt"], add_special_tokens=False)["input_ids"]) for r in r42]
            plen_all[fmt] = pl
            out["prompt_ranges"][f"{fam}|{fmt}"] = [min(pl), max(pl)]
            band = near = both = 0
            for s in range(42, 47):
                for i, x in enumerate(details("math500", fam, fmt, s)):
                    t = x["completion_tokens"]; cap = MAX_LEN - pl[i]
                    band += t >= cap - TOL; near += t >= 32500; both += (t >= cap - TOL) and t >= 32500
            out["cap_band_math500"][f"{fam}|{fmt}"] = {"cap_band": band, "near_cap": near, "both": both}
        rb, ra = raw("math500", fam, "BF16", 42), raw("math500", fam, "AWQ-4", 42)
        out["awq_suffix_equal"][fam] = {"string": sum(y["full_prompt"] + "<think>\n" == x["full_prompt"] for x, y in zip(rb, ra)),
                                        "token_ids": sum(tk(x["full_prompt"], add_special_tokens=False)["input_ids"] == tk(y["full_prompt"], add_special_tokens=False)["input_ids"] + tk.encode("<think>\n", add_special_tokens=False) for x, y in zip(rb, ra)), "n": len(rb)}
        # exact cap on every benchmark (rows that could be near the cap)
        for bench in BENCH:
            ex = out["exact_cap"].setdefault(bench, {"rows_ge_31900": 0, "exact_cap": 0, "within_2_of_cap": 0, "above_cap": 0, "ge_32500": 0, "ge_32500_and_exact": 0})
            for fmt in FMT:
                for s in BENCH[bench][3]:
                    for r, x in zip(raw(bench, fam, fmt, s), details(bench, fam, fmt, s)):
                        t = x["completion_tokens"]
                        if t < 31900:
                            continue
                        cap = MAX_LEN - len(tk(r["full_prompt"], add_special_tokens=False)["input_ids"])
                        ex["rows_ge_31900"] += 1; ex["exact_cap"] += t == cap; ex["within_2_of_cap"] += (t != cap and abs(t - cap) <= 2)
                        ex["above_cap"] += t > cap; ex["ge_32500"] += t >= 32500; ex["ge_32500_and_exact"] += (t >= 32500 and t == cap)
        if full:
            n = eq = 0
            for bench in BENCH:
                for fmt in FMT:
                    for s in BENCH[bench][3]:
                        for r, x in zip(raw(bench, fam, fmt, s), details(bench, fam, fmt, s)):
                            n += 1; eq += len(tk(r["generated_text"], add_special_tokens=False)["input_ids"]) == x["completion_tokens"]
            out.setdefault("completion_tokens_recount", {})[fam] = {"completions": n, "equal": eq}
    return out


# ------------------------------------------------------------------------------------------------ 3-5 inference
def inference_checks() -> dict:
    out = {}
    W = weights(500)
    # --- serving aggregation
    raws = REPO / "results" / "measured_serving_confirmation" / "raw"
    ms = {"Qwen": "Qwen-7B", "Llama": "Llama-8B"}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 500, (B, 500))
    out["campaign_len_cost"] = {}
    for fam in FAM:
        T = {f: tok(fam, f) for f in FMT}; A = {f: acc(fam, f) for f in FMT}
        for cond in "AB":
            reps = {f: [json.loads(p.read_text()) for p in sorted(raws.glob(f"{ms[fam]}_{f}_rep*_cond{cond}.json"))] for f in FMT}
            for variant in ("mean_of_ratios", "total_tokens_over_total_time"):
                ts = {f: (float(np.mean([r["output_tokens_per_second"] for r in reps[f]])) if variant == "mean_of_ratios" else sum(r["total_output_tokens"] for r in reps[f]) / sum(r["elapsed_seconds"] for r in reps[f])) for f in FMT}
                sec = {f: float(T[f].mean() / ts[f] / A[f].mean()) for f in FMT}
                cost = np.stack([T[f].mean(0)[idx].mean(1) / ts[f] / A[f].mean(0)[idx].mean(1) for f in FMT], 1)
                p1 = np.bincount(cost.argmin(1), minlength=4) / B
                out["campaign_len_cost"][f"{fam}|{cond}|{variant}"] = {"tok_s": ts, "seconds_per_correct": sec, "p_rank1": dict(zip(FMT, [float(x) for x in p1]))}
    # --- paired conditional-minus-placebo (BF16-correct) intervals
    out["bf16_correct_minus_placebo"] = {}
    for fam in FAM:
        T = {f: tok(fam, f) for f in FMT}; C = {f: acc(fam, f).astype(bool) for f in FMT}
        pn = np.zeros(500); pd = np.zeros(500)
        for s, t in itertools.permutations(range(5), 2):
            pn += np.where(C["BF16"][s], T["BF16"][t] - T["BF16"][s], 0); pd += C["BF16"][s]
        for q in FMT[1:]:
            cn = np.zeros(500); cd = np.zeros(500)
            for s in range(5):
                cn += np.where(C["BF16"][s], T[q][s] - T["BF16"][s], 0); cd += C["BF16"][s]
            est = cn.sum() / cd.sum() - pn.sum() / pd.sum(); bs = (W @ cn) / (W @ cd) - (W @ pn) / (W @ pd)
            out["bf16_correct_minus_placebo"][f"{fam}|{q}"] = {"estimate": float(est), "ci95": ci95(bs)}
    # --- maj@5 change in the gap
    out["maj5_gap_change"] = {}
    for fam in FAM:
        Bm = acc(fam, "BF16")
        for q in FMT[1:]:
            Aq = acc(fam, q); dp = Aq.mean(0) - Bm.mean(0); dm = (Aq.sum(0) >= 3).astype(float) - (Bm.sum(0) >= 3).astype(float)
            out["maj5_gap_change"][f"{fam}|{q}"] = {"pass1_gap_pp": float(100 * dp.mean()), "maj5_gap_pp": float(100 * dm.mean()), "change_pp": float(100 * (dm - dp).mean()), "change_ci95_pp": [100 * x for x in ci95((W @ (dm - dp)) / 500)]}
    # --- Llama FP8 Condition A resampling
    sub = json.loads((REPO / "results/measured_serving_confirmation/condition_a_subset.json").read_text()); rows = np.array([int(x["campaign_index"]) for x in sub])
    T = tok("Llama", "FP8"); measured = float(np.mean([json.loads(p.read_text())["mean_output_tokens_per_req"] for p in sorted(raws.glob("Llama-8B_FP8_rep*_condA.json"))]))
    g = np.random.default_rng(0); draws = np.array([T[g.integers(0, 5, len(rows)), rows].mean() for _ in range(20000)])
    out["llama_fp8_condA"] = {"measured_mean_tokens_per_query": measured, "campaign_mean_on_subset_items": float(T[:, rows].mean()), "per_seed_subset_means": [float(x) for x in T[:, rows].mean(1)],
                              "max_possible_mean_from_stored_seeds": float(T[:, rows].max(0).mean()), "draw_max": float(draws.max()), "fraction_of_draws_below_measured": float((draws < measured).mean())}
    # --- CV of first three repeats (population vs sample SD)
    r3 = [json.loads(p.read_text())["output_tokens_per_second"] for p in sorted(raws.glob("Llama-8B_FP8_rep*_condA.json"))][:3]
    out["llama_fp8_condA_cv"] = {"population_sd": float(np.std(r3) / np.mean(r3)), "sample_sd": float(np.std(r3, ddof=1) / np.mean(r3))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="also recount every stored completion_tokens from text (slow)")
    ap.add_argument("--out", type=Path, default=REPO / "results/reports/audit_checks.json")
    a = ap.parse_args()
    rep = {"tokenizers": tokenizer_checks(a.full), "inference": inference_checks()}
    a.out.write_text(json.dumps(rep, indent=1) + "\n")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
