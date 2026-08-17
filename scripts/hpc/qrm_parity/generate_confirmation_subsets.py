#!/usr/bin/env python3
"""Generate and validate balanced deterministic subsets for serving confirmation."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "results" / "measured_serving_confirmation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load HF MATH-500 dataset
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

# 2. Load Campaign prompt mapping
camp_path = REPO_ROOT / "outputs-hpc-campaign-2026-08-14" / "qrm_official_DeepSeek-R1-Distill-Qwen-7B_math500_n500_seed42.json"
with open(camp_path, "r", encoding="utf-8") as f:
    camp = json.load(f)

# Build lookup by stripped problem text
hf_lookup = {item["problem"].strip(): item for item in ds}

all_aligned = []
for idx, c in enumerate(camp):
    fp = c["full_prompt"]
    user_content = fp.split("<｜User｜>")[1].split("\n\nPlease reason")[0].strip()
    hf_item = hf_lookup.get(user_content)
    if not hf_item:
        for p, item in hf_lookup.items():
            if p in user_content or user_content in p:
                hf_item = item
                break
    assert hf_item is not None, f"Failed match for campaign index {idx}"

    lvl_raw = hf_item["level"]
    lvl_int = int(str(lvl_raw).replace("Level", "").strip())

    entry = {
        "campaign_index": idx,
        "math500_problem_index": idx,
        "level": lvl_int,
        "subject": hf_item["subject"],
        "problem": hf_item["problem"],
        "solution": hf_item["solution"],
        "answer": hf_item["answer"],
        "full_prompt": fp,
    }
    all_aligned.append(entry)

assert len(all_aligned) == 500, f"Expected 500 aligned items, got {len(all_aligned)}"

# 3. Deterministic selection with fixed seed 20260817
rng = random.Random(20260817)
by_level = {lvl: [] for lvl in range(1, 6)}
for item in all_aligned:
    by_level[item["level"]].append(item)

# Condition A: Exactly 4 per level (20 items total)
cond_a_items = []
for lvl in range(1, 6):
    chosen = rng.sample(by_level[lvl], 4)
    for subset_idx, it in enumerate(chosen):
        item_copy = dict(it)
        item_copy["subset_a_index"] = len(cond_a_items)
        cond_a_items.append(item_copy)

# Condition B: Exactly 20 per level (100 items total)
cond_b_items = []
for lvl in range(1, 6):
    chosen = rng.sample(by_level[lvl], 20)
    for subset_idx, it in enumerate(chosen):
        item_copy = dict(it)
        item_copy["subset_b_index"] = len(cond_b_items)
        cond_b_items.append(item_copy)

# 4. Strict assertions
assert len(cond_a_items) == 20
assert Counter(x["level"] for x in cond_a_items) == {1: 4, 2: 4, 3: 4, 4: 4, 5: 4}
for it in cond_a_items:
    assert it["problem"] in it["full_prompt"], f"Problem text not in full_prompt for A idx {it['subset_a_index']}"

assert len(cond_b_items) == 100
assert Counter(x["level"] for x in cond_b_items) == {1: 20, 2: 20, 3: 20, 4: 20, 5: 20}
for it in cond_b_items:
    assert it["problem"] in it["full_prompt"], f"Problem text not in full_prompt for B idx {it['subset_b_index']}"

# 5. Write to disk
file_a = OUT_DIR / "condition_a_subset.json"
file_b = OUT_DIR / "condition_b_subset.json"

file_a.write_text(json.dumps(cond_a_items, indent=2) + "\n", encoding="utf-8")
file_b.write_text(json.dumps(cond_b_items, indent=2) + "\n", encoding="utf-8")

print(f"Successfully generated and validated:")
print(f"  Condition A subset (20 items, 4/level): {file_a}")
print(f"  Condition B subset (100 items, 20/level): {file_b}")
