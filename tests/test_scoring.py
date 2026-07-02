"""Golden tests for correctness scoring (canonical evaluation layer)."""

from src.evaluation.correctness.scoring import (
    maj_at_k,
    majority_vote_answer,
    score_gpqa_item,
    score_gsm8k_item,
    score_item,
    score_math_item,
    summarize_failure_rates,
    summarize_scored_rows,
)


def test_score_math_correct_boxed():
    out = score_math_item("\\boxed{42}", "Reasoning... \\boxed{42}")
    assert out["correct"] is True
    assert out["pred_answer"] == "42"


def test_score_math_wrong_answer():
    out = score_math_item("\\boxed{42}", "Answer: \\boxed{7}")
    assert out["correct"] is False


def test_score_math_malformed_no_boxed():
    out = score_math_item("\\boxed{1}", "no answer here")
    assert out["correct"] is False
    assert out["answer_parse_success"] is False


def test_score_gsm8k_numeric():
    gold = "Jan has 3 apples. #### 42"
    out = score_gsm8k_item(gold, "The answer is 42")
    assert out["correct"] is True


def test_score_gsm8k_comma_number():
    gold = "Total #### 1,234"
    out = score_gsm8k_item(gold, "Final: 1234")
    assert out["correct"] is True


def test_score_gpqa_letter():
    out = score_gpqa_item("B", "I choose (B) as the answer.")
    assert out["correct"] is True


def test_score_item_routes_gpqa():
    row = {"task": "gpqa_diamond", "gold_letter": "A", "completion": "Answer: A"}
    assert score_item(row)["correct"] is True


def test_maj_at_k_majority():
    assert maj_at_k([True, True, False]) is True
    assert maj_at_k([True, False, False]) is False


def test_majority_vote_tie_picks_one():
    assert majority_vote_answer(["a", "b", "a", "b"]) in ("a", "b")


def test_summarize_failure_rates_truncation():
    rows = [
        {"pred_answer": "1", "completion": "x", "truncated": True, "finish_reason": "length"},
        {"pred_answer": None, "completion": "", "truncated": False},
    ]
    rates = summarize_failure_rates(rows)
    assert rates["truncation_rate"] == 0.5
    assert rates["empty_completion_rate"] == 0.5


def test_summarize_scored_rows_golden():
    rows = [
        {"correct": True, "latency_sec": 2.0, "peak_vram_gb": 10.0},
        {"correct": False, "latency_sec": 4.0, "peak_vram_gb": 12.0},
    ]
    summary = summarize_scored_rows(rows)
    assert summary["n"] == 2
    assert summary["pass_at_1"] == 0.5
    assert "pass_at_1_ci95_low" in summary
