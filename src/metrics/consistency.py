"""Sample-consistency confidence metrics.

Adapted from Calibrating-LLMs-with-Consistency/source/calibrate/consistency.py
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, List, Optional, Sequence


def calculate_consistency(answers: Sequence[str], method: str = "agree_percent") -> float:
    """Return confidence in [0, 1] from multiple sampled answers."""
    frequency: Counter[str] = Counter()
    for answer in answers:
        answer_str = str(answer)
        frequency[answer_str] += 1

    if len(answers) == 0:
        return 0.0

    if "[invalid]" in frequency and len(frequency) > 1:
        del frequency["[invalid]"]

    if method == "entropy":
        n_valid = sum(frequency.values())
        probs = [f / n_valid for f in frequency.values()]
        entropy = -sum(p * math.log2(p) for p in probs)
        denom = math.log2(len(frequency)) if len(frequency) > 1 else 0.0
        normalized_entropy = entropy / denom if denom > 0 else 0.0
        consistency = 1.0 - normalized_entropy
    elif method == "agree_percent":
        consistency = max(frequency.values()) / len(answers)
    elif method == "fsd":
        sorted_freq = sorted(frequency.values(), reverse=True)
        if len(sorted_freq) > 1:
            diff = sorted_freq[0] - sorted_freq[1]
        else:
            diff = len(answers)
        consistency = diff / len(answers)
    else:
        raise ValueError(f"Unknown consistency method: {method}")

    return float(min(1.0, max(0.0, consistency)))


def majority_answer(answers: Sequence[str]) -> Optional[str]:
    if not answers:
        return None
    counts = Counter(str(a) for a in answers)
    top, _ = counts.most_common(1)[0]
    if top == "[invalid]" and len(counts) > 1:
        top = counts.most_common(2)[1][0]
    for answer in answers:
        if str(answer) == top:
            return answer
    return top
