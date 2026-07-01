#!/usr/bin/env python3
"""Preflight: verify decoding YAML → vLLM SamplingParams (sober-reasoning + QRM repro)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runners.config_utils import load_cell_config, load_decoding_from_file
from src.runners.sampling_utils import (
    QRM_REPRO_REFERENCE,
    SOBER_REFERENCE,
    build_sampling_params_dict,
    verify_decoding_for_vllm,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify decoding params before HPC rerun.")
    parser.add_argument(
        "--decoding-config",
        default="configs/decoding/repro_qrm.yaml",
        help="Decoding YAML to verify.",
    )
    parser.add_argument(
        "--cell-config",
        default=None,
        help="Optional cell JSON (uses cell seed + merged decoding).",
    )
    parser.add_argument(
        "--no-require-repetition-penalty",
        action="store_true",
        help="Do not warn if repetition_penalty is missing.",
    )
    args = parser.parse_args()

    if args.cell_config:
        cell = load_cell_config(args.cell_config)
        decoding = cell["decoding"]
        seed = int(cell.get("seed", decoding.get("seed", 0)))
        print(f"Cell: {cell.get('cell_id')} seed={seed}")
    else:
        decoding = load_decoding_from_file(args.decoding_config)
        seed = int(decoding.get("seed", 0))
        print(f"Decoding file: {args.decoding_config} seed={seed}")

    ok, messages = verify_decoding_for_vllm(
        decoding,
        seed,
        require_repetition_penalty=not args.no_require_repetition_penalty,
    )
    for msg in messages:
        print(msg)

    params = build_sampling_params_dict(decoding, seed)
    print("\nSamplingParams dict (vLLM):")
    for k, v in params.items():
        print(f"  {k}: {v}")

    print("\nReference protocols:")
    print(f"  QRM repro (inference.py): temp={QRM_REPRO_REFERENCE['temperature']}, "
          f"top_p={QRM_REPRO_REFERENCE['top_p']}, max_tokens={QRM_REPRO_REFERENCE['max_new_tokens']}")
    print(f"  sober-reasoning (main.py): temp={SOBER_REFERENCE['temperature']}, "
          f"top_p={SOBER_REFERENCE['top_p']}, repetition_penalty=optional")

    if not ok:
        print("\nVERIFY FAILED — fix decoding before HPC rerun.")
        sys.exit(1)
    print("\nVERIFY OK")


if __name__ == "__main__":
    main()
