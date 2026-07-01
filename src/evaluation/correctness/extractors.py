"""Answer extractors — canonical re-export."""

from src.extraction.gpqa_extractor import extract_gpqa_letter, normalize_gpqa_letter
from src.extraction.math_extractor import (
    extract_boxed,
    extract_gold_answer,
    normalize_answer,
    normalize_completion_text,
)

__all__ = [
    "extract_boxed",
    "extract_gold_answer",
    "extract_gpqa_letter",
    "normalize_answer",
    "normalize_completion_text",
    "normalize_gpqa_letter",
]
