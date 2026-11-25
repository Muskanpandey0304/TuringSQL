import pytest

from turingsql.safety import QueryTimeoutError, UnsafeSQLError, check_sql, execute_readonly


@pytest.mark.parametrize("sql", [
    "SELECT 1",
    "SELECT state, COUNT(*) FROM dim_customer GROUP BY state",
    "WITH x AS (SELECT 1 AS a) SELECT a FROM x",
    "SELECT 1 UNION SELECT 2",
])
def test_allowed(sql):
    check_sql(sql)


@pytest.mark.parametrize("sql", [
    "",
    "DELETE FROM fact_orders",
    "DROP TABLE fact_orders",
    "INSERT INTO t VALUES (1)",
    "UPDATE t SET a = 1",
    "SELECT 1; DROP TABLE t",
    "ATTACH DATABASE 'x.db' AS x",
    "PRAGMA table_info(t)",
    "CREATE TABLE t (a INT)",
    "SELEC 1 FROM",
    # model output with a backslash-escaped quote makes sqlglot raise TokenError; it must count as a wrong
    # answer, not crash a whole evaluation run (this killed a 9 h Kaggle run)
    "SELECT Code FROM sets WHERE name = 'Ancestor\\'s Chosen')",
])
def test_rejected(sql):
    with pytest.raises(UnsafeSQLError):
        check_sql(sql)


def test_execute_returns_rows(tiny_db):
    rows = execute_readonly(tiny_db, "SELECT COUNT(*) FROM fact_orders WHERE is_late = 1")
    assert rows == [(2,)]


def test_row_cap(tiny_db):
    assert len(execute_readonly(tiny_db, "SELECT * FROM fact_orders", max_rows=2)) == 2


def test_timeout(tiny_db):
    slow = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 100000000) "
        "SELECT COUNT(*) FROM c"
    )
    with pytest.raises(QueryTimeoutError):
        execute_readonly(tiny_db, slow, timeout_s=0.2)


def test_unsafe_never_executes(tiny_db):
    with pytest.raises(UnsafeSQLError):
        execute_readonly(tiny_db, "DELETE FROM fact_orders")
    assert execute_readonly(tiny_db, "SELECT COUNT(*) FROM fact_orders") == [(3,)]
