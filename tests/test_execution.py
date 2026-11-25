import pytest

from turingsql.eval.execution import results_match, score, values_equal

GOLD_LATE = "SELECT COUNT(*) FROM fact_orders WHERE is_late = 1"


# ---------- values ----------

@pytest.mark.parametrize("gold,pred", [
    (1, 1.0), (2.5, 2.5), (None, None), ("SP", "SP"),
    (4.294117, 4.29),          # model rounded to 2 decimals
    (8.130723, 8.1),           # rounded to 1 decimal
    (4.294117, 4.3),           # rounded to 1 decimal
    (0.1 + 0.2, 0.3),          # float noise
])
def test_values_equal(gold, pred):
    assert values_equal(gold, pred)


@pytest.mark.parametrize("gold,pred", [
    (4.29, 4),                 # whole number never matches a non-whole value
    (8.13, 0.0813),            # fraction vs percentage
    (4.294117, 4.28),          # wrong at 2 decimals
    (4.294117, 4.29415),       # 5 decimals: not treated as a rounding, and outside float noise (7e-6)
    (None, 0), ("1", "01"),
])
def test_values_not_equal(gold, pred):
    assert not values_equal(gold, pred)


# ---------- results ----------

def test_row_order_ignored_unless_required():
    gold = [("SP", 3), ("RJ", 2)]
    pred = [("RJ", 2), ("SP", 3)]
    assert results_match(gold, pred, ordered=False, allow_extra_columns=False)
    assert not results_match(gold, pred, ordered=True, allow_extra_columns=False)


def test_column_permutation_accepted():
    assert results_match([("SP", 3), ("RJ", 2)], [(3, "SP"), (2, "RJ")], ordered=False, allow_extra_columns=False)


def test_extra_column_strict_vs_relaxed():
    gold = [("SP", 3)]
    pred = [("SP", "Southeast", 3)]
    assert not results_match(gold, pred, ordered=False, allow_extra_columns=False)
    assert results_match(gold, pred, ordered=False, allow_extra_columns=True)


def test_missing_column_or_rows_fails():
    assert not results_match([("SP", 3)], [("SP",)], ordered=False, allow_extra_columns=True)
    assert not results_match([("SP", 3), ("RJ", 2)], [("SP", 3)], ordered=False, allow_extra_columns=True)


def test_duplicates_count_as_multiset():
    assert not results_match([(1,), (1,), (2,)], [(1,), (2,), (2,)], ordered=False, allow_extra_columns=False)


def test_tolerant_multiset_with_rounding():
    gold = [("a", 1.23456), ("b", 2.34567)]
    pred = [("b", 2.35), ("a", 1.23)]
    assert results_match(gold, pred, ordered=False, allow_extra_columns=False)


# ---------- end to end on a real SQLite file ----------

def test_gold_vs_gold(tiny_db):
    s = score(tiny_db, GOLD_LATE, GOLD_LATE)
    assert s.correct and s.correct_relaxed and s.error is None


def test_equivalent_sql_and_fenced_reply(tiny_db):
    reply = "```sql\nSELECT SUM(is_late) FROM fact_orders;\n```"
    assert score(tiny_db, GOLD_LATE, reply).correct


def test_wrong_answer(tiny_db):
    s = score(tiny_db, GOLD_LATE, "SELECT COUNT(*) FROM fact_orders")
    assert not s.correct and s.error is None


@pytest.mark.parametrize("pred,error", [
    ("", "empty_sql"),
    ("DELETE FROM fact_orders", "unsafe"),
    ("SELECT nope FROM fact_orders", "exec_error"),
])
def test_errors_reported(tiny_db, pred, error):
    s = score(tiny_db, GOLD_LATE, pred)
    assert not s.correct and s.error == error


def test_timeout_reported(tiny_db):
    slow = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 100000000) "
            "SELECT COUNT(*) FROM c")
    s = score(tiny_db, GOLD_LATE, slow, timeout_s=0.2)
    assert s.error == "timeout"
