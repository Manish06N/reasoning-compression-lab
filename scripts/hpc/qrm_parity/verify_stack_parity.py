#!/usr/bin/env python3
"""Print QRM vs our-stack parity checklist (no GPU required)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.runners.config_utils import load_cell_config, load_decoding_from_file
from src.runners.sampling_utils import build_sampling_params_dict, verify_decoding_for_vllm
from src.runners.vllm_serving import build_llm_init_kwargs, load_serving_defaults


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify QRM stack parity for a cell config.")
    parser.add_argument(
        "--cell-config",
        default="configs/cells/diag_qwen7b_bf16_math500_seed42_n10_parity.json",
    )
    args = parser.parse_args()

    cell = load_cell_config(args.cell_config)
    decoding = cell["decoding"]
    model_cfg = cell["model"]
    seed = int(cell["seed"])
    serving = load_serving_defaults()

    print("=== QRM stack parity checklist ===")
    print(f"cell_id: {cell['cell_id']}")
    print(f"prompt_profile: {cell.get('prompt_profile')}")
    print(f"prompt_template: {cell['task'].get('prompt_template_file')}")
    print()

    ok, messages = verify_decoding_for_vllm(
        decoding, seed, require_repetition_penalty=False
    )
    print("Decoding → SamplingParams:")
    for line in messages:
        print(f"  {line}")
    params = build_sampling_params_dict(decoding, seed)
    print(f"  logprobs in SamplingParams: {'logprobs' in params}")
    print()

    llm_kwargs = build_llm_init_kwargs(
        cell["model_path"], model_cfg, seed=seed, serving_defaults=serving
    )
    checks = [
        ("LLM seed", llm_kwargs.get("seed"), seed),
        ("gpu_memory_utilization", llm_kwargs.get("gpu_memory_utilization"), 0.9),
        ("enable_prefix_caching", llm_kwargs.get("enable_prefix_caching"), False),
        ("enable_chunked_prefill", llm_kwargs.get("enable_chunked_prefill"), False),
        ("enforce_eager", llm_kwargs.get("enforce_eager"), True),
        ("max_model_len", llm_kwargs.get("max_model_len"), decoding.get("max_model_len")),
    ]
    print("LLM() kwargs vs QRM inference.py:")
    all_ok = ok
    for name, observed, expected in checks:
        status = "OK" if observed == expected else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  [{status}] {name}: {observed!r} (expected {expected!r})")
    print()
    print("Overall:", "PASS" if all_ok else "FAIL")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()