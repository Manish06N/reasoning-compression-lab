"""Extract final answers from math reasoning completions.

Adapted from sober-reasoning/lighteval_tasks.py (MATH_QUERY_TEMPLATE + boxed extraction).
"""

from __future__ import annotations

import re
from typing import Optional

BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")
FINAL_ANSWER_PATTERN = re.compile(
    r"final answer is:\s*(?:\\boxed\{([^}]*)\}|([^\n\.]+))",
    re.IGNORECASE,
)


def extract_boxed(text: str) -> Optional[str]:
    matches = BOXED_PATTERN.findall(text)
    if not matches:
        match = FINAL_ANSWER_PATTERN.search(text)
        if match:
            return (match.group(1) or match.group(2) or "").strip()
        return None
    return matches[-1].strip()


def extract_gold_answer(solution: str) -> Optional[str]:
    return extract_boxed(solution)


def normalize_answer(text: Optional[str]) -> str:
    if text is None:
        return ""
    cleaned = text.strip()
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace("\\,", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned.lower()
