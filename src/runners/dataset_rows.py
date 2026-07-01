"""Build per-problem rows from HuggingFace dataset examples (MATH, GPQA, GSM8K)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any


def output_root_for(out_path: Path) -> Path | None:
    """Infer archive root (parent of raw/) when output lives under .../raw/."""
    if out_path.parent.name == "raw":
        return out_path.parent.parent
    return None


def get_field(example: dict, *names: str) -> str:
    for name in names:
        if name in example and example[name] is not None:
            return str(example[name])
    return ""


def prepare_example_row(
    example: dict,
    task: dict,
    cell: dict,
    global_i: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Return (prompt_fields, row_base) for one dataset example."""
    task_name = task["task_name"]
    if task_name.startswith("gpqa"):
        question = get_field(example, "Question", "question", "problem")
        correct = get_field(example, "Correct Answer", "correct_answer", "answer")
        distractors = [
            get_field(example, "Incorrect Answer 1", "incorrect_answer_1"),
            get_field(example, "Incorrect Answer 2", "incorrect_answer_2"),
            get_field(example, "Incorrect Answer 3", "incorrect_answer_3"),
        ]
        choices = [(correct, "correct")] + [(d, "incorrect") for d in distractors]
        rng = random.Random(int(cell.get("seed", 0)) * 1_000_003 + global_i)
        rng.shuffle(choices)
        letters = "ABCD"
        prompt_fields: dict[str, str] = {"question": question}
        gold_letter = None
        for letter, (choice, kind) in zip(letters, choices):
            prompt_fields[letter.lower()] = choice
            if kind == "correct":
                gold_letter = letter
        row_base = {
            "id": example.get("Record ID", example.get("unique_id", str(global_i))),
            "problem": question,
            "choices": {letter: prompt_fields[letter.lower()] for letter in letters},
            "gold_letter": gold_letter,
            "gold_answer": correct,
        }
        return prompt_fields, row_base

    problem_field = task.get("problem_field", "problem")
    solution_field = task.get("solution_field", "solution")
    problem = str(example[problem_field])
    return {"question": problem.strip()}, {
        "id": example.get("unique_id", str(global_i)),
        "problem": problem,
        "gold_solution": example[solution_field],
        "subject": example.get("subject"),
        "level": example.get("level"),
    }
