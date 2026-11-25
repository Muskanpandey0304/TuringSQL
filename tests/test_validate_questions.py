"""The question validator's checks must themselves be right; a silent tie check would pass anything."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_prep"))

import validate_olist_questions as v  # noqa: E402

pytestmark = pytest.mark.skipif(not v.DB.is_file(), reason="run data_prep/build_olist_marts.py first")


def test_tie_detected_at_limit():
    assert v.limit_tie("SELECT state FROM dim_customer ORDER BY state LIMIT 1") is not None


def test_no_tie_reported_for_distinct_keys():
    assert v.limit_tie("SELECT state, COUNT(*) AS n FROM dim_customer GROUP BY state ORDER BY n DESC LIMIT 3") is None
    assert v.limit_tie("SELECT COUNT(*) FROM fact_orders") is None


def test_validator_flags_bad_questions():
    bad = [
        {"id": "X1", "level": "easy", "question": "q1", "gold_sql": "DELETE FROM fact_orders", "order_matters": False},
        {"id": "X2", "level": "easy", "question": "q2", "gold_sql": "SELECT 1 WHERE 0", "order_matters": False},
        {"id": "X3", "level": "easy", "question": "q3", "gold_sql": "SELECT 1", "order_matters": True},
        {"id": "X3", "level": "nope", "question": "q4", "gold_sql": "SELECT 1", "order_matters": False},
    ]
    errors, _ = v.validate(bad)
    text = "\n".join(errors)
    assert "X1: gold SQL failed" in text
    assert "X2: gold SQL returned no rows" in text
    assert "X3: order_matters is true" in text
    assert "duplicate id X3" in text
    assert "unknown level" in text


def test_split_is_stratified_and_deterministic():
    qs = [{"id": f"{lvl[0]}{i:02d}", "level": lvl} for lvl, n in
          (("easy", 30), ("medium", 40), ("hard", 35), ("definition", 15)) for i in range(n)]
    dev, test = v.split(qs)
    assert (len(dev), len(test)) == (40, 80)
    assert {q["id"] for q in dev}.isdisjoint({q["id"] for q in test})
    assert v.split(qs) == (dev, test)
