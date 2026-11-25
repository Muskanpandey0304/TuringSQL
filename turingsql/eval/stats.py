"""Statistics for comparing configs on the same questions.

Ported from AdaptiveRAG (src/eval.py at commit 62fb23a, MIT licence, same author) so both projects
report confidence intervals and significance the same way.
"""
from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(values: Sequence[float], n_resamples: int = 2000,
                 seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% percentile-bootstrap interval of per-question 0/1 outcomes.

    Resamples questions with replacement. Fixed seed so a report is reproducible.
    """
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    low = means[int(0.025 * (n_resamples - 1))]
    high = means[int(0.975 * (n_resamples - 1))]
    return _mean(values), low, high


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value for two configs scored on the same questions.

    `only_a` = questions A got right and B got wrong; `only_b` the reverse. Under "no real difference"
    each disagreement is a fair coin flip, so this is a binomial test with p = 0.5.
    Returns 1.0 when the configs never disagree.
    """
    disagreements = only_a + only_b
    if disagreements == 0:
        return 1.0
    tail = sum(math.comb(disagreements, k) for k in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / 2**disagreements)


def paired_comparison(correct_a: Mapping[str, bool], correct_b: Mapping[str, bool]) -> dict:
    """Compare two configs' per-question correctness, keyed by question id."""
    shared = sorted(set(correct_a) & set(correct_b))
    only_a = sum(1 for i in shared if correct_a[i] and not correct_b[i])
    only_b = sum(1 for i in shared if correct_b[i] and not correct_a[i])
    return {"n": len(shared), "only_a": only_a, "only_b": only_b, "p_value": mcnemar_exact(only_a, only_b)}
