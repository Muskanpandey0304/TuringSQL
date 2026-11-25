import pytest

from turingsql.eval.stats import bootstrap_ci, mcnemar_exact, paired_comparison


def test_bootstrap_bounds_and_reproducible():
    values = [1.0] * 30 + [0.0] * 20
    mean, low, high = bootstrap_ci(values)
    assert mean == pytest.approx(0.6)
    assert low < mean < high
    assert bootstrap_ci(values) == (mean, low, high)


def test_bootstrap_edge_cases():
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0, 1.0)


def test_mcnemar_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 1) == pytest.approx(0.21875)   # 2 * (1 + 6) / 64
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert mcnemar_exact(3, 3) == 1.0


def test_paired_comparison_uses_shared_ids_only():
    a = {"q1": True, "q2": True, "q3": False, "q4": True}
    b = {"q1": False, "q2": True, "q3": True}
    r = paired_comparison(a, b)
    assert (r["n"], r["only_a"], r["only_b"]) == (3, 1, 1)
