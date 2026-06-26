"""GPQA multiple-choice answer extraction.

Adapted from sober-reasoning/lighteval_tasks.py (gpqa_metric / letter extraction).
"""

from __future__ import annotations

import re
from typing import Optional

ANSWER_LINE = re.compile(r"Answer:\s*\$?([A-Da-d])\$?", re.IGNORECASE)
LETTER_PATTERN = re.compile(r"\b([A-Da-d])\b")


def extract_gpqa_letter(text: str) -> Optional[str]:
    match = ANSWER_LINE.search(text)
    if match:
        return match.group(1).upper()
    letters = LETTER_PATTERN.findall(text[-200:])
    if letters:
        return letters[-1].upper()
    return None


def normalize_gpqa_letter(text: Optional[str]) -> str:
    if not text:
        return ""
    return text.strip().upper()
