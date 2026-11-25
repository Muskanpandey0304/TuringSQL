import pytest

from turingsql.ask import AskOlist, add_limit, suggest_chart
from turingsql.safety import UnsafeSQLError


def test_add_limit():
    sql, added = add_limit("SELECT a FROM t", 5)
    assert added and sql.upper().endswith("LIMIT 5")
    assert add_limit("SELECT a FROM t LIMIT 3", 5) == ("SELECT a FROM t LIMIT 3", False)
    with pytest.raises(UnsafeSQLError):
        add_limit("DROP TABLE t")


def test_suggest_chart():
    assert suggest_chart(["month", "rev"], [("2018-01", 1.0), ("2018-02", 2.0)]) == \
        {"type": "line", "x": "month", "y": ["rev"]}
    assert suggest_chart(["state", "n"], [("SP", 3), ("RJ", 2)])["type"] == "bar"
    assert suggest_chart(["n"], [(1,), (2,)]) is None                    # one column
    assert suggest_chart(["state", "n"], [("SP", 3)]) is None           # one row
    assert suggest_chart(["a", "b"], [("x", "y"), ("z", "w")]) is None  # nothing numeric


def test_ask_end_to_end(tiny_db):
    seen = {}

    def gen(messages):
        seen["prompt"] = messages[-1]["content"]
        return "```sql\nSELECT customer_unique_id, COUNT(*) AS n FROM fact_orders GROUP BY 1\n```"

    app = AskOlist(tiny_db, gen, retrieve=lambda q: ["Order: one row in fact_orders."])
    ans = app.ask("orders per customer?")
    assert ans.error is None and ans.limited
    assert ans.columns == ["customer_unique_id", "n"]
    assert sorted(ans.rows) == [("c1", 2), ("c2", 1)]
    assert ans.chart == {"type": "bar", "x": "customer_unique_id", "y": ["n"]}
    assert "### Definitions" in seen["prompt"] and ans.definitions


def test_ask_refuses_writes_and_reports_errors(tiny_db):
    app = AskOlist(tiny_db, lambda m: "DELETE FROM fact_orders")
    ans = app.ask("delete everything")
    assert ans.error.startswith("not run") and ans.rows == []
    assert app.ask("x", sql_override="SELECT COUNT(*) FROM fact_orders").rows == [(3,)]   # data untouched
    assert "no such column" in app.ask("x", sql_override="SELECT nope FROM fact_orders").error

    def boom(m):
        raise ConnectionError("server down")

    assert "server down" in AskOlist(tiny_db, boom).ask("q").error
