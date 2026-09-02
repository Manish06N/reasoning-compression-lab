#!/usr/bin/env python3
"""Fail if paper/main.tex is missing frozen campaign numbers."""

from __future__ import annotations

import argparse
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEX = os.path.join(REPO, "paper", "main.tex")

NEEDLES = [
    "88 checkpoint",
    "56{,}408",
    "vLLM~0.7.0",
    "Marlin W8A16",
    "$-2.76$",
    "$-5.56$",
    "$-1.57$",
    "6.3$--$6.9",
    "Holm-6 family",
    "Holm-18 joint sensitivity",
    "$-45.9$",
    "tested community AWQ",
    "not a property of bit-width alone",
    "75 of 198",
    "0.109",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        print("Pass --check.", file=__import__("sys").stderr)
        return 2
    text = open(TEX, encoding="utf-8").read()
    missing = [n for n in NEEDLES if n not in text]
    if missing:
        print("ERROR: manuscript missing frozen needles:", file=__import__("sys").stderr)
        for n in missing:
            print(f"  {n}", file=__import__("sys").stderr)
        return 1
    print(f"OK: {len(NEEDLES)} frozen manuscript needles present in paper/main.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
