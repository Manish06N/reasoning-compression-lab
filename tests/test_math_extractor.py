"""Tests for MATH answer extraction."""

from src.extraction.math_extractor import extract_boxed, normalize_completion_text, normalize_answer


def test_extract_simple_boxed():
    assert extract_boxed(r"The answer is \boxed{42}") == "42"


def test_extract_nested_boxed():
    assert extract_boxed(r"Therefore, $\boxed{\frac{1}{2}}$") == r"\frac{1}{2}"


def test_extract_prompt_format():
    text = "Therefore, the final answer is: $\\boxed{(3, \\frac{\\pi}{2})}$. I hope it is correct"
    assert extract_boxed(text) == r"(3, \frac{\pi}{2})"


def test_normalize_sentencepiece_markers():
    raw = "FirstĠanswerĊĊ\\boxed{42}"
    assert normalize_completion_text(raw) == "First answer\n\n\\boxed{42}"
    assert extract_boxed(raw) == "42"


def test_normalize_answer_strips_spaces():
    assert normalize_answer("(3, pi/2)") == normalize_answer("(3,pi/2)")


def test_extract_skips_unclosed_trailing_boxed():
    text = r"So \boxed{5} ... later \boxed{\frac{1}{2"
    assert extract_boxed(text) == "5"
