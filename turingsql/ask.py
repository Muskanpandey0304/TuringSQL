"""The "Ask Olist" pipeline behind the demo app (PROJECT_PLAN.md section 9).

    question -> glossary definitions -> prompt -> model (any OpenAI-compatible server, e.g. TuringServe)
             -> SQL -> safety check -> read-only execution (LIMIT added) -> rows + a suggested chart

The model's SQL is always returned to the caller and shown; nothing runs that is not a single read-only query.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from turingsql.prompt import build_messages, extract_sql
from turingsql.safety import QueryTimeoutError, UnsafeSQLError, check_sql, connect_readonly
from turingsql.schema import serialize_schema

DEFAULT_LIMIT = 1000
Generate = Callable[[list[dict[str, str]]], str]
Retrieve = Callable[[str], list[str]]


@dataclass
class Answer:
    question: str
    sql: str = ""
    raw_output: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    definitions: list[str] = field(default_factory=list)
    chart: dict | None = None
    error: str | None = None
    limited: bool = False
    timings: dict[str, float] = field(default_factory=dict)


def add_limit(sql: str, limit: int = DEFAULT_LIMIT) -> tuple[str, bool]:
    """Add LIMIT to the outer query if it has none. Returns (sql, added)."""
    tree = check_sql(sql)
    if tree.args.get("limit") is not None:
        return sql, False
    return tree.limit(limit, copy=True).sql(dialect="sqlite"), True


def _is_dateish(values: list) -> bool:
    return bool(values) and all(isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", v) for v in values)


def _is_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def suggest_chart(columns: list[str], rows: list[tuple]) -> dict | None:
    """Line chart for (date-like, number...), bar chart for (category, number...) with 2-50 rows, else none."""
    if len(rows) < 2 or len(columns) < 2 or len(rows) > 50:
        return None
    first = [r[0] for r in rows]
    numeric = [i for i in range(1, len(columns)) if all(_is_number(r[i]) or r[i] is None for r in rows)]
    if not numeric:
        return None
    kind = "line" if _is_dateish(first) else "bar"
    if kind == "bar" and not all(isinstance(v, (str, int)) for v in first):
        return None
    return {"type": kind, "x": columns[0], "y": [columns[i] for i in numeric]}


class AskOlist:
    def __init__(self, db_path: Path, generate: Generate, retrieve: Retrieve | None = None,
                 timeout_s: float = 10.0, limit: int = DEFAULT_LIMIT):
        self.db_path = Path(db_path)
        self.schema = serialize_schema(self.db_path)
        self.generate = generate
        self.retrieve = retrieve
        self.timeout_s = timeout_s
        self.limit = limit

    def ask(self, question: str, sql_override: str | None = None) -> Answer:
        ans = Answer(question=question)
        t0 = time.perf_counter()
        if sql_override is None:
            if self.retrieve is not None:
                ans.definitions = self.retrieve(question)
            t1 = time.perf_counter()
            ans.timings["retrieval_s"] = t1 - t0
            try:
                ans.raw_output = self.generate(build_messages(self.schema, question, ans.definitions or None))
            except Exception as e:  # noqa: BLE001 - shown to the user, not raised
                ans.error = f"model request failed: {type(e).__name__}: {e}"
                return ans
            ans.timings["generation_s"] = time.perf_counter() - t1
            ans.sql = extract_sql(ans.raw_output)
        else:
            ans.sql = extract_sql(sql_override)
        return self.run(ans)

    def run(self, ans: Answer) -> Answer:
        t0 = time.perf_counter()
        try:
            sql, ans.limited = add_limit(ans.sql, self.limit)
        except UnsafeSQLError as e:
            ans.error = f"not run: {e}"
            return ans
        deadline = time.monotonic() + self.timeout_s
        conn = connect_readonly(self.db_path)
        try:
            conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
            cur = conn.execute(sql)
            ans.columns = [d[0] for d in cur.description or []]
            ans.rows = [tuple(r) for r in cur.fetchall()]
        except Exception as e:  # noqa: BLE001
            if "interrupted" in str(e).lower():
                e = QueryTimeoutError(f"query exceeded {self.timeout_s}s")
            ans.error = f"{type(e).__name__}: {e}"
            return ans
        finally:
            conn.close()
        ans.timings["execution_s"] = time.perf_counter() - t0
        ans.chart = suggest_chart(ans.columns, ans.rows)
        return ans
