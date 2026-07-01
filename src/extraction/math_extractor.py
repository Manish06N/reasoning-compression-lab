"""Extract final answers from math reasoning completions.

Adapted from sober-reasoning/lighteval_tasks.py and lm-eval last_boxed_only_string.
"""

from __future__ import annotations

import re
from typing import Optional

FINAL_ANSWER_LINE = re.compile(
    r"final answer is:\s*(.+?)(?:\.\s*I hope it is correct|\.\s*$|\s*$)",
    re.IGNORECASE | re.DOTALL,
)


def normalize_completion_text(text: str) -> str:
    """Repair vLLM 0.8.x Llama completions that leak SentencePiece markers."""
    if not text:
        return text
    if "\u0120" not in text and "Ġ" not in text and "\u010a" not in text and "Ċ" not in text:
        return text
    return (
        text.replace("\u0120", " ")
        .replace("Ġ", " ")
        .replace("\u010a", "\n")
        .replace("Ċ", "\n")
    )


def _match_braced_span(text: str, start: int) -> Optional[str]:
    i = start
    depth = 0
    started = False
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            started = True
        elif ch == "}":
            depth -= 1
            if started and depth == 0:
                return text[start : i + 1]
            if depth < 0:
                return None
        i += 1
    return None


def _boxed_candidates(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for token in ("\\boxed", "\\fbox"):
        start = 0
        while True:
            idx = text.find(token, start)
            if idx < 0:
                break
            out.append((idx, token))
            start = idx + len(token)
    out.sort(key=lambda item: item[0], reverse=True)
    return out


def _last_boxed_span(text: str) -> Optional[str]:
    for idx, token in _boxed_candidates(text):
        if token == "\\boxed" and text.startswith("\\boxed ", idx):
            return text[idx:].split("$", 1)[0].strip()
        span = _match_braced_span(text, idx)
        if span:
            return span
    return None


def _unwrap_boxed(span: str) -> Optional[str]:
    if span.startswith("\\boxed{"):
        return span[7:-1].strip()
    if span.startswith("\\fbox{"):
        return span[6:-1].strip()
    if span.startswith("\\boxed "):
        return span[7:].strip()
    return None


def extract_boxed(text: str) -> Optional[str]:
    text = normalize_completion_text(text)
    span = _last_boxed_span(text)
    if span:
        return _unwrap_boxed(span)

    match = FINAL_ANSWER_LINE.search(text)
    if match:
        tail = match.group(1).strip()
        span = _last_boxed_span(tail)
        if span:
            return _unwrap_boxed(span)
        cleaned = tail.strip().strip("$").strip()
        if cleaned:
            return cleaned
    return None


def extract_gold_answer(solution: str) -> Optional[str]:
    return extract_boxed(solution)


def normalize_answer(text: Optional[str]) -> str:
    if text is None:
        return ""
    cleaned = normalize_completion_text(text.strip())
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace("\\,", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned.lower()
