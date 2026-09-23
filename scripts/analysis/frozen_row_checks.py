#!/usr/bin/env python3
"""CPU checks on frozen campaign JSON. No GPU. No new measured table cells.

Prints quant-correct bootstrap intervals, serving-repeat min/max,
unboxed x near-cap counts, and, when the public MATH-500 split and
tokenizers are available, a level audit and a prompt-length cap recount.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATH = ROOT / "results/math500"
GSM = ROOT / "results/gsm8k"
GPQA = ROOT / "results/gpqa"
SERVING = ROOT / "results/measured_serving_confirmation/raw"
N_BOOT = 10_000
BOOT_SEED = 0
NEAR_CAP = 32_500
MODEL_LEN = 32_768

FAMILIES = {
    "DeepSeek-R1-Distill-Qwen-7B": ("Qwen-7B", "BF16"),
    "DeepSeek-R1-Distill-Qwen-7B-FP8": ("Qwen-7B", "FP8"),
    "DeepSeek-R1-Distill-Qwen-7B-AWQ-4": ("Qwen-7B", "AWQ-4"),
    "DeepSeek-R1-Distill-Qwen-7B-GPTQ-4": ("Qwen-7B", "GPTQ-4"),
    "DeepSeek-R1-Distill-Llama-8B": ("Llama-8B", "BF16"),
    "DeepSeek-R1-Distill-Llama-8B-FP8": ("Llama-8B", "FP8"),
    "DeepSeek-R1-Distill-Llama-8B-AWQ-4": ("Llama-8B", "AWQ-4"),
    "DeepSeek-R1-Distill-Llama-8B-GPTQ-4": ("Llama-8B", "GPTQ-4"),
}
PASS1 = {
    ("Qwen-7B", "BF16"): 94.00,
    ("Qwen-7B", "FP8"): 94.40,
    ("Qwen-7B", "AWQ-4"): 93.12,
    ("Qwen-7B", "GPTQ-4"): 93.48,
    ("Llama-8B", "BF16"): 89.24,
    ("Llama-8B", "FP8"): 89.52,
    ("Llama-8B", "AWQ-4"): 86.48,
    ("Llama-8B", "GPTQ-4"): 88.92,
}
ORDER = [
    ("Qwen-7B", "BF16"),
    ("Qwen-7B", "FP8"),
    ("Qwen-7B", "AWQ-4"),
    ("Qwen-7B", "GPTQ-4"),
    ("Llama-8B", "BF16"),
    ("Llama-8B", "FP8"),
    ("Llama-8B", "AWQ-4"),
    ("Llama-8B", "GPTQ-4"),
]


def load_math() -> dict[tuple[str, str], dict[int, list[dict]]]:
    cells: dict[tuple[str, str], dict[int, list[dict]]] = {}
    for path in sorted(MATH.glob("*.json")):
        stem = path.name.split("_math500_")[0]
        fam_fmt = FAMILIES[stem]
        seed = int(path.name.split("seed")[1].split(".")[0])
        payload = json.loads(path.read_text())
        cells.setdefault(fam_fmt, {})[seed] = payload["details"]
    return cells


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def quant_correct_ci(cells: dict) -> None:
    print("quant-correct item bootstrap (same stream as the BF16-correct column)")
    contrasts = [
        (("Qwen-7B", "BF16"), ("Qwen-7B", "FP8")),
        (("Qwen-7B", "BF16"), ("Qwen-7B", "AWQ-4")),
        (("Qwen-7B", "BF16"), ("Qwen-7B", "GPTQ-4")),
        (("Llama-8B", "BF16"), ("Llama-8B", "FP8")),
        (("Llama-8B", "BF16"), ("Llama-8B", "AWQ-4")),
        (("Llama-8B", "BF16"), ("Llama-8B", "GPTQ-4")),
    ]
    for base_key, other_key in contrasts:
        base = cells[base_key]
        other = cells[other_key]
        seeds = sorted(base)
        n_items = len(base[seeds[0]])
        item_quant: list[list[float]] = [[] for _ in range(n_items)]
        all_quant: list[float] = []
        for i in range(n_items):
            for seed in seeds:
                b_row = base[seed][i]
                o_row = other[seed][i]
                bc = b_row.get("extractive_match", 0.0) == 1.0
                oc = o_row.get("extractive_match", 0.0) == 1.0
                if not oc:
                    continue
                rec = float(o_row.get("completion_tokens") or 0) - float(
                    b_row.get("completion_tokens") or 0
                )
                item_quant[i].append(rec)
                all_quant.append(rec)
        rng = random.Random(BOOT_SEED)
        boot: list[float] = []
        for _ in range(N_BOOT):
            pool: list[float] = []
            for _i in range(n_items):
                pool.extend(item_quant[rng.randrange(n_items)])
            if pool:
                boot.append(sum(pool) / len(pool))
        mean = sum(all_quant) / len(all_quant)
        lo, hi = percentile(boot, 2.5), percentile(boot, 97.5)
        print(
            f"  {other_key[0]} {other_key[1]}: n={len(all_quant)} "
            f"mean={mean:.1f} CI=[{lo:.0f},{hi:.0f}] "
            f"excludes0={lo > 0 or hi < 0}"
        )


def cross_tab(folder: Path, tag: str) -> None:
    print(f"unboxed x near-cap ({tag})")
    totals = {"unboxed": 0, "near": 0, "both": 0, "rows": 0}
    for path in sorted(folder.glob("*.json")):
        payload = json.loads(path.read_text())
        unboxed = near = both = 0
        for row in payload["details"]:
            tok = int(row.get("completion_tokens") or 0)
            boxed = bool(row.get("boxed", True))
            is_near = tok >= NEAR_CAP
            if not boxed:
                unboxed += 1
            if is_near:
                near += 1
            if (not boxed) and is_near:
                both += 1
        totals["unboxed"] += unboxed
        totals["near"] += near
        totals["both"] += both
        totals["rows"] += len(payload["details"])
        if tag == "MATH-500":
            stem = path.name.split("_math500_")[0]
            fam, fmt = FAMILIES[stem]
            print(f"  {fam} {fmt} {path.name[-12:-5]}: unboxed={unboxed} near={near} both={both}")
    print(
        f"  TOTAL {tag}: rows={totals['rows']} unboxed={totals['unboxed']} "
        f"near={totals['near']} both={totals['both']}"
    )


def serving_minmax() -> None:
    print("serving repeat min-max; dollar uses fixed campaign pass@1")
    groups: dict[tuple, list[dict]] = {}
    for path in SERVING.glob("*_cond*.json"):
        payload = json.loads(path.read_text())
        key = (payload["model"], payload["format"], payload["condition"][0])
        groups.setdefault(key, []).append(payload)
    for fam, fmt in ORDER:
        for cond in ("A", "B"):
            reps = groups[(fam, fmt, cond)]
            tps = [r["output_tokens_per_second"] for r in reps]
            gpu = [r["gpu_seconds_per_query"] for r in reps]
            acc = PASS1[(fam, fmt)] / 100.0
            dollars = [g * 1.5 / 3600 / acc for g in gpu]
            print(
                f"  {fam} {fmt} {cond}: n={len(reps)} "
                f"tok/s {min(tps):.2f}-{max(tps):.2f} "
                f"gpu {min(gpu):.2f}-{max(gpu):.2f} "
                f"$ {min(dollars):.4f}-{max(dollars):.4f}"
            )


def level_and_cap() -> None:
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        print(f"level/cap skipped: {exc}")
        return
    math = load_dataset("HuggingFaceH4/MATH-500", split="test")
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    levels = [str(row["level"]) for row in math]
    print("MATH-500 level counts", {lv: levels.count(lv) for lv in sorted(set(levels))})
    qwen_tok = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", trust_remote_code=True
    )
    llama_tok = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", trust_remote_code=True
    )

    def prompt_len(tokenizer, text: str) -> int:
        user = (
            text
            + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
        )
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return len(tokenizer.encode(rendered, add_special_tokens=False))

    math_prompt = {
        "Qwen-7B": [prompt_len(qwen_tok, row["problem"]) for row in math],
        "Llama-8B": [prompt_len(llama_tok, row["problem"]) for row in math],
    }
    gsm_prompt = {
        "Qwen-7B": [prompt_len(qwen_tok, row["question"]) for row in gsm],
        "Llama-8B": [prompt_len(llama_tok, row["question"]) for row in gsm],
    }
    print(
        "MATH prompt tokens Qwen",
        min(math_prompt["Qwen-7B"]),
        max(math_prompt["Qwen-7B"]),
    )
    print(
        "MATH prompt tokens Llama",
        min(math_prompt["Llama-8B"]),
        max(math_prompt["Llama-8B"]),
    )

    print("level-2 vs level-5 pass@1 and unboxed, Qwen/Llama BF16 seed-mean")
    for fam in ("Qwen-7B", "Llama-8B"):
        for fmt in ("BF16", "FP8", "AWQ-4", "GPTQ-4"):
            by_level: dict[str, list[tuple[int, int]]] = {str(i): [0, 0] for i in range(1, 6)}
            unboxed_level = {str(i): [0, 0] for i in range(1, 6)}
            details = load_math()[(fam, fmt)]
            for rows in details.values():
                for i, row in enumerate(rows):
                    lv = levels[i]
                    by_level[lv][1] += 1
                    unboxed_level[lv][1] += 1
                    if row.get("extractive_match", 0.0) == 1.0:
                        by_level[lv][0] += 1
                    if not row.get("boxed", True):
                        unboxed_level[lv][0] += 1
            bits = []
            for lv in ("1", "2", "3", "4", "5"):
                correct, n = by_level[lv]
                ub, un = unboxed_level[lv]
                bits.append(f"L{lv} {100 * correct / n:.2f}% ub {ub}/{un}")
            print(f"  {fam} {fmt}: " + " | ".join(bits))

    print("cap recount: completion_tokens >= 32768 - chat-template prompt tokens")
    for folder, tag, prompts in (
        (MATH, "MATH-500", math_prompt),
        (GSM, "GSM8K", gsm_prompt),
    ):
        for path in sorted(folder.glob("*.json")):
            if tag == "MATH-500":
                stem = path.name.split("_math500_")[0]
            else:
                stem = path.name.split("_gsm8k_")[0]
            fam, fmt = FAMILIES[stem]
            payload = json.loads(path.read_text())
            hits = near_and_hit = unboxed_hit = 0
            for i, row in enumerate(payload["details"]):
                cap = MODEL_LEN - prompts[fam][i]
                tok = int(row.get("completion_tokens") or 0)
                if tok >= cap:
                    hits += 1
                    if tok >= NEAR_CAP:
                        near_and_hit += 1
                    if not row.get("boxed", True):
                        unboxed_hit += 1
            if hits:
                print(
                    f"  {tag} {fam} {fmt} {path.name}: hits={hits} "
                    f"also_near={near_and_hit} unboxed_hits={unboxed_hit}"
                )


def main() -> None:
    cells = load_math()
    quant_correct_ci(cells)
    cross_tab(MATH, "MATH-500")
    cross_tab(GSM, "GSM8K")
    cross_tab(GPQA, "GPQA")
    serving_minmax()
    level_and_cap()


if __name__ == "__main__":
    main()
