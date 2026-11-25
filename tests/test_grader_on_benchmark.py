"""The grader, checked on the real Olist benchmark.

1. Every gold query scored against itself must be correct (no false negatives on identical SQL).
2. Every gold query scored against every *other* question's gold query must be wrong under the strict
   rule (no false positives from an over-lenient grader).
3. Under the relaxed rule (extra columns allowed), the only cross matches are the known superset pairs:
   answers that contain another question's full answer plus extra columns.
"""
from __future__ import annotations

import itertools

import pytest

from turingsql.benchmark import OLIST_DIR, load_questions
from turingsql.eval.execution import results_match, run_sql, score

DB = OLIST_DIR / "olist_marts.db"
pytestmark = pytest.mark.skipif(not DB.is_file(), reason="run data_prep/build_olist_marts.py first")

# (question, other question whose answer contains this answer plus extra columns)
KNOWN_SUPERSETS = {
    ("M03", "H02"),   # top-10 categories by total  <  same rows + rank + cumulative share
    ("M05", "M28"),   # avg delivery days by region <  same rows + avg estimated days
}


@pytest.fixture(scope="module")
def gold():
    questions = load_questions("dev") + load_questions("test")
    out = {}
    for q in questions:
        res = run_sql(DB, q["gold_sql"], timeout_s=60)
        assert res.rows is not None, (q["id"], res.error, res.message)
        out[q["id"]] = (q, res.rows)
    return out


def test_all_120_questions_loaded(gold):
    assert len(gold) == 120


def test_gold_vs_gold_is_always_correct(gold):
    wrong = [qid for qid, (q, rows) in gold.items()
             if not score(DB, q["gold_sql"], q["gold_sql"], q["order_matters"], gold_rows=rows).correct]
    assert wrong == []


@pytest.mark.slow   # about 14,000 pairs, roughly a minute
def test_other_questions_answers_are_wrong(gold):
    strict, relaxed = [], []
    for (a, (qa, ra)), (b, (_, rb)) in itertools.permutations(gold.items(), 2):
        if results_match(ra, rb, qa["order_matters"], allow_extra_columns=False):
            strict.append((a, b))
        if results_match(ra, rb, qa["order_matters"], allow_extra_columns=True):
            relaxed.append((a, b))
    assert strict == []
    assert set(relaxed) == KNOWN_SUPERSETS
