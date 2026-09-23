#!/usr/bin/env python3
"""Q1-revision analyses from frozen campaign outputs (CPU only, read-only inputs).

Adds, without changing any frozen result:
  1. MATH-500 per-difficulty-level pass@1 and paired deltas vs BF16. Pipeline rows
     are joined to the public dataset by exact question text taken from the stored
     campaign prompt, not by position.
  2. A per-item effective-cap count: completions within a small tolerance of
     32,768 minus that item's own prompt length, set against the >=32,500 proxy.
  3. The chat-template audit: which cells end the generation prompt with <think>\\n,
     and whether GPQA prompts, including the answer-choice order, are identical
     across checkpoints within each family and seed.

Output: results/reports/q1_revision_analyses.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FAMILIES = {"Qwen-7B": "DeepSeek-R1-Distill-Qwen-7B", "Llama-8B": "DeepSeek-R1-Distill-Llama-8B"}
FORMATS = ["BF16", "FP8", "AWQ-4", "GPTQ-4"]
SEEDS = {"math500": [42, 43, 44, 45, 46], "gsm8k": [42, 43, 44], "gpqa": [42, 43, 44]}
RAW = {
    "math500": ("outputs-hpc-campaign-2026-08-14", "MATH-500.jsonl"),
    "gsm8k": ("outputs-hpc-breadth-gsm8k-2026-08-15", "GSM8K.jsonl"),
    "gpqa": ("outputs-hpc-breadth-gpqa-2026-08-16", "GPQA-Diamond.jsonl"),
}
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MAX_MODEL_LEN = 32768
CAP_TOLERANCE = 16  # re-tokenised counts differ from vLLM's by up to about +/-15 tokens
USER, ASSIST = "<｜User｜>", "<｜Assistant｜>"
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
B, RNG_SEED = 10_000, 0


def cell_name(family: str, fmt: str) -> str:
    return FAMILIES[family] if fmt == "BF16" else f"{FAMILIES[family]}-{fmt}"


def raw_rows(bench: str, family: str, fmt: str, seed: int) -> list[dict]:
    root, fname = RAW[bench]
    return json.loads((REPO / root / "inference" / f"{cell_name(family, fmt)}-seed{seed}" / fname).read_text())


def compact_rows(bench: str, family: str, fmt: str, seed: int) -> list[dict]:
    n = {"math500": 500, "gsm8k": 1319, "gpqa": 198}[bench]
    tag = {"gpqa": "gpqadiamond"}.get(bench, bench)
    path = REPO / "results" / bench / f"{cell_name(family, fmt)}_{tag}_n{n}_seed{seed}.json"
    return json.loads(path.read_text())["details"]


def question(full_prompt: str) -> str:
    q = full_prompt.split(USER, 1)[1].split(ASSIST, 1)[0]
    return q[: -len(SUFFIX)] if q.endswith(SUFFIX) else q


def boot_ci(x: np.ndarray, rng: np.random.Generator) -> list[float]:
    idx = rng.integers(0, len(x), (B, len(x)))
    lo, hi = np.quantile(x[idx].mean(1), [0.025, 0.975])
    return [round(float(lo) * 100, 2), round(float(hi) * 100, 2)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", str(REPO / "hf_cache")))
    ap.add_argument("--out", type=Path, default=REPO / "results/reports/q1_revision_analyses.json")
    args = ap.parse_args()
    os.environ["HF_HOME"] = args.hf_home
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", revision=MATH500_REVISION)
    by_q = {row["problem"].strip(): (i, int(row["level"])) for i, row in enumerate(ds)}
    rng = np.random.default_rng(RNG_SEED)
    report: dict = {"inputs": {"math500_revision": MATH500_REVISION, "bootstrap": {"B": B, "seed": RNG_SEED},
                               "cap_tolerance_tokens": CAP_TOLERANCE}}

    # ---- 1 + 2: MATH-500 join, levels, effective cap
    levels: dict = {}
    cap: dict = {}
    for family in FAMILIES:
        tok = AutoTokenizer.from_pretrained(REPO / "models" / FAMILIES[family])
        corr, tokens, level_of, plen = {}, {}, None, None
        for fmt in FORMATS:
            C, T = [], []
            for seed in SEEDS["math500"]:
                raw, det = raw_rows("math500", family, fmt, seed), compact_rows("math500", family, fmt, seed)
                assert len(raw) == len(det) == 500
                if level_of is None:
                    joined = [by_q[question(r["full_prompt"]).strip()] for r in raw]
                    assert len({j[0] for j in joined}) == 500, "question join is not one-to-one"
                    level_of = np.array([lv for _, lv in joined])
                    plen = np.array([len(tok(r["full_prompt"], add_special_tokens=False)["input_ids"]) for r in raw])
                C.append([float(d["extractive_match"]) for d in det])
                T.append([int(d["completion_tokens"]) for d in det])
            corr[fmt], tokens[fmt] = np.array(C), np.array(T)
        report.setdefault("join", {})[family] = "500/500 pipeline rows matched to distinct dataset items by exact question text"
        levels[family] = {"n_items_per_level": {str(lv): int((level_of == lv).sum()) for lv in range(1, 6)}}
        for fmt in FORMATS:
            item = corr[fmt].mean(0)
            levels[family][fmt] = {str(lv): round(float(item[level_of == lv].mean()) * 100, 2) for lv in range(1, 6)}
        for fmt in FORMATS[1:]:
            d = corr[fmt].mean(0) - corr["BF16"].mean(0)
            levels[family][f"{fmt}_minus_BF16"] = {
                str(lv): {"delta_pp": round(float(d[level_of == lv].mean()) * 100, 2),
                          "ci95_pp": boot_ci(d[level_of == lv], rng)} for lv in range(1, 6)}
        eff_cap = MAX_MODEL_LEN - plen
        cap[family] = {"prompt_tokens_range": [int(plen.min()), int(plen.max())]}
        for fmt in FORMATS:
            T = tokens[fmt]
            in_band = T >= (eff_cap[None, :] - CAP_TOLERANCE)
            near = T >= 32_500
            cap[family][fmt] = {"cap_band": int(in_band.sum()), "near_cap_proxy": int(near.sum()),
                                "proxy_and_band": int((in_band & near).sum()),
                                "band_not_proxy": int((in_band & ~near).sum()),
                                "proxy_not_band": int((near & ~in_band).sum()),
                                "cap_band_by_level": {str(lv): int(in_band[:, level_of == lv].sum()) for lv in range(1, 6)}}
    report["math500_levels"] = levels
    report["math500_effective_cap"] = cap

    # ---- 3: template audit and GPQA pairing
    template: dict = {}
    gpqa_pairing: Counter = Counter()
    for bench in ("math500", "gsm8k", "gpqa"):
        for family in FAMILIES:
            prompts = defaultdict(dict)
            for fmt in FORMATS:
                ends, opens = 0, 0
                for seed in SEEDS[bench]:
                    raw = raw_rows(bench, family, fmt, seed)
                    ends += sum(r["full_prompt"].endswith("<think>\n") for r in raw)
                    opens += sum(r["generated_text"].startswith("<think>\n") for r in raw)
                    for i, r in enumerate(raw):
                        p = r["full_prompt"]
                        prompts[seed].setdefault(i, {})[fmt] = p if p.endswith("<think>\n") else p + "<think>\n"
                n = sum(len(raw_rows(bench, family, fmt, s)) for s in SEEDS[bench][:1]) * len(SEEDS[bench])
                template[f"{bench}|{family}|{fmt}"] = {"rows": n, "prompt_ends_with_think": ends,
                                                        "completion_opens_with_think": opens}
            for seed, rows in prompts.items():
                for i, per_fmt in rows.items():
                    key = "identical_after_suffix_normalisation" if len(set(per_fmt.values())) == 1 else "differ"
                    gpqa_pairing[f"{bench}|{family}|{key}"] += 1
    report["chat_template_audit"] = template
    report["prompt_identity_across_checkpoints"] = dict(gpqa_pairing)
    tk_q = AutoTokenizer.from_pretrained(REPO / "models" / FAMILIES["Qwen-7B"])
    tk_l = AutoTokenizer.from_pretrained(REPO / "models" / FAMILIES["Llama-8B"])
    report["think_suffix_tokens"] = {"Qwen-7B": len(tk_q("<think>\n", add_special_tokens=False)["input_ids"]),
                                     "Llama-8B": len(tk_l("<think>\n", add_special_tokens=False)["input_ids"])}

    payload = json.dumps(report, indent=2) + "\n"
    args.out.write_text(payload)
    print(json.dumps({"out": str(args.out), "sha256": hashlib.sha256(payload.encode()).hexdigest()[:16]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
