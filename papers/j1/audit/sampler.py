"""Manual audit sampler for J1 extraction gate (V8.2 §6.7)."""

from __future__ import annotations

import random
from typing import List, Sequence


def sample_audit_rows(
    rows: Sequence[dict],
    *,
    n: int = 50,
    seed: int = 0,
    oversample_failures: bool = True,
) -> List[dict]:
    if not rows:
        return []
    rng = random.Random(seed)
    failures = [r for r in rows if not r.get("correct") or not r.get("answer_parse_success")]
    successes = [r for r in rows if r not in failures]
    out: List[dict] = []
    if oversample_failures and failures:
        take_f = min(len(failures), max(n // 2, n - min(len(successes), n // 2)))
        out.extend(rng.sample(failures, take_f))
    remaining = n - len(out)
    pool = successes if remaining <= len(successes) else rows
    if remaining > 0 and pool:
        out.extend(rng.sample(pool, min(remaining, len(pool))))
    return out[:n]
