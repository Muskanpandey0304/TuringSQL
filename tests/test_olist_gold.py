"""The Olist benchmark database must reproduce the numbers published in the Olist project README.

If one of these fails, either the marts copy is wrong or a definition drifted. Fix that before
trusting any gold answer. Skipped when data/olist/olist_marts.db has not been built (e.g. in CI).
"""
from pathlib import Path

import pytest

from turingsql.safety import execute_readonly

DB = Path(__file__).resolve().parents[1] / "data" / "olist" / "olist_marts.db"
pytestmark = pytest.mark.skipif(not DB.is_file(), reason="run data_prep/build_olist_marts.py first")


def one(sql: str):
    rows = execute_readonly(DB, sql)
    assert len(rows) == 1
    return rows[0]


def test_headline_revenue_orders_customers():
    revenue, orders, customers = one(
        "SELECT SUM(price), COUNT(DISTINCT order_id), COUNT(DISTINCT customer_unique_id) "
        "FROM fact_order_items WHERE is_revenue_item = 1"
    )
    assert round(revenue / 1e6, 2) == 13.18          # README: R$13.18M
    assert orders == 96_203                           # README: 96,203 delivered orders
    assert customers == 93_096                        # README: 93,096 customers


def test_repeat_rate():
    (rate,) = one(
        "WITH c AS (SELECT customer_unique_id, COUNT(*) AS n FROM fact_orders "
        "WHERE is_revenue_order = 1 GROUP BY customer_unique_id) "
        "SELECT 100.0 * SUM(n >= 2) / COUNT(*) FROM c"
    )
    assert round(rate, 1) == 3.0                      # README: 3.0%


def test_late_rate():
    (rate,) = one("SELECT 100.0 * AVG(is_late) FROM fact_orders WHERE is_revenue_order = 1")
    assert round(rate, 1) == 8.1                      # README: 8.1% of orders late


def test_review_late_vs_on_time():
    rows = dict(execute_readonly(
        DB,
        "SELECT is_late, AVG(review_score) FROM fact_orders WHERE is_revenue_order = 1 GROUP BY is_late",
    ))
    assert round(rows[1], 2) == 2.57                  # README: late orders 2.57 stars
    assert round(rows[0], 2) == 4.29                  # README: on time 4.29 stars


def test_average_order_value():
    (aov,) = one(
        "SELECT SUM(price) / COUNT(DISTINCT order_id) FROM fact_order_items WHERE is_revenue_item = 1"
    )
    assert 135 <= aov <= 139                          # README: about R$137
