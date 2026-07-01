from src.extraction.math_extractor import extract_boxed, normalize_answer, normalize_generated_text


def test_extract_boxed_nested_fraction():
    assert extract_boxed(r"The answer is \boxed{\frac{1}{2}}.") == r"\frac{1}{2}"


def test_extract_boxed_uses_last_box():
    assert extract_boxed(r"\boxed{1} then corrected to \boxed{2}") == "2"


def test_extract_boxed_unclosed_tail():
    assert extract_boxed(r"Final: \boxed{\frac{3}{4}") == r"\frac{3}{4}"


def test_extract_boxed_nested_set():
    assert extract_boxed(r"\boxed{\left\{1, \frac{2}{3}\right\}}") == r"\left\{1, \frac{2}{3}\right\}"


def test_sentencepiece_artifact_normalization():
    assert normalize_generated_text("TheĠanswerĊis") == "The answer\nis"


def test_normalize_answer_removes_spacing():
    assert normalize_answer(r" $ \frac { 1 } { 2 } $ ") == r"\frac{1}{2}"
