"""Run model-written SQL safely.

Two independent layers, so one mistake does not open the database:
1. `check_sql`: sqlglot must parse exactly one read-only query (SELECT / WITH ... SELECT / UNION ...),
   with no write, DDL, ATTACH or PRAGMA nodes anywhere inside it.
2. `execute_readonly`: the connection itself is opened with mode=ro, has a wall-clock timeout and a
   row cap.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import sqlglot
from sqlglot import exp

from turingsql.schema import connect_readonly

_FORBIDDEN_NAMES = (
    "Insert", "Update", "Delete", "Merge", "Drop", "Create", "Alter", "TruncateTable",
    "Attach", "Detach", "Pragma", "Command", "Transaction", "Commit", "Rollback", "Set", "Use",
)
_FORBIDDEN: tuple[type[exp.Expression], ...] = tuple(
    t for t in (getattr(exp, n, None) for n in _FORBIDDEN_NAMES)
    if isinstance(t, type) and issubclass(t, exp.Expression)
)


class UnsafeSQLError(ValueError):
    """The SQL is not a single read-only query."""


class QueryTimeoutError(RuntimeError):
    """The query ran longer than the allowed time."""


def check_sql(sql: str) -> exp.Query:
    """Raise UnsafeSQLError unless `sql` is exactly one read-only query. Returns the parsed tree."""
    if not sql or not sql.strip():
        raise UnsafeSQLError("empty SQL")
    try:
        trees = [t for t in sqlglot.parse(sql, read="sqlite") if t is not None]
    except sqlglot.errors.SqlglotError as e:  # ParseError and TokenError (e.g. a backslash-escaped quote)
        raise UnsafeSQLError(f"does not parse: {e}") from e
    if len(trees) != 1:
        raise UnsafeSQLError(f"expected one statement, got {len(trees)}")
    tree = trees[0]
    if not isinstance(tree, exp.Query):
        raise UnsafeSQLError(f"not a query: {type(tree).__name__}")
    for node in tree.walk():
        if isinstance(node, _FORBIDDEN):
            raise UnsafeSQLError(f"forbidden statement inside query: {type(node).__name__}")
    return tree


def execute_readonly(db_path: str | Path, sql: str, timeout_s: float = 10.0,
                     max_rows: int = 10_000) -> list[tuple]:
    """Check, then run `sql` on a read-only connection. Returns at most `max_rows` rows."""
    check_sql(sql)
    deadline = time.monotonic() + timeout_s
    conn = connect_readonly(db_path)
    try:
        # Called every N SQLite VM instructions; a non-zero return aborts the query.
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        try:
            cur = conn.execute(sql)
            return cur.fetchmany(max_rows)
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e).lower():
                raise QueryTimeoutError(f"query exceeded {timeout_s}s") from e
            raise
    finally:
        conn.close()
