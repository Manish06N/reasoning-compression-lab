"""Extract final answers from math reasoning completions.

Adapted from sober-reasoning/lighteval_tasks.py (MATH_QUERY_TEMPLATE + boxed extraction).
"""

from __future__ import annotations

import re
from typing import Optional

FINAL_ANSWER_PATTERN = re.compile(
    r"final answer is:\s*(?:\\boxed\{([^}]*)\}|([^\n\.]+))",
    re.IGNORECASE,
)


def normalize_generated_text(text: str) -> str:
    """Normalize tokenizer artifacts observed in stored completions."""
    if not text:
        return ""
    return (
        text.replace("Ġ", " ")
        .replace("Ċ", "\n")
        .replace("▁", " ")
        .replace("<0x0A>", "\n")
    )


def _extract_braced_content(text: str, open_brace: int) -> Optional[str]:
    depth = 0
    content_start = open_brace + 1
    i = open_brace
    while i < len(text):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:i].strip()
        i += 1
    tail = text[content_start:].strip()
    return tail or None


def extract_boxed(text: str) -> Optional[str]:
    text = normalize_generated_text(text)
    starts = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    for start in reversed(starts):
        answer = _extract_braced_content(text, start + len(r"\boxed"))
        if answer:
            return answer

    match = FINAL_ANSWER_PATTERN.search(text)
    if match:
        return (match.group(1) or match.group(2) or "").strip()
    return None


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
