"""Execution accuracy: run predicted and gold SQL, compare the results, not the SQL text.

Rules (see docs/DECISIONS.md, decision 12):
- Rows are compared as a multiset; only when `order_matters` is true must the order match too.
- Columns may come back in a different order (a permutation of the gold columns is accepted).
- `strict`: same number of columns as gold. `relaxed`: the prediction may have extra columns, as long as
  some choice of its columns matches the gold result exactly.
- Numbers: int and float compare by value (1 == 1.0) with a tiny tolerance for float noise. A predicted
  float with 1-4 decimals also matches when it equals the gold value rounded to that many decimals
  (the model rounded its answer). A whole number never matches a non-whole gold value.
"""
from __future__ import annotations

import itertools
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from turingsql.prompt import extract_sql
from turingsql.safety import QueryTimeoutError, UnsafeSQLError, execute_readonly

Row = tuple
MAX_ROWS = 10_000
MAX_PERMUTED_COLUMNS = 6       # 720 permutations; wider results must match in gold column order
MAX_ROWS_FOR_TOLERANT_MATCH = 2_000


@dataclass
class ExecResult:
    rows: list[Row] | None = None
    error: str | None = None       # "unsafe" | "timeout" | "exec_error" | "empty_sql" | "too_many_rows"
    message: str = ""


@dataclass
class Score:
    correct: bool
    correct_relaxed: bool
    error: str | None = None
    message: str = ""
    n_pred_rows: int | None = None
    details: dict = field(default_factory=dict)


def run_sql(db_path: str | Path, sql: str, timeout_s: float = 10.0) -> ExecResult:
    if not sql.strip():
        return ExecResult(error="empty_sql")
    try:
        rows = execute_readonly(db_path, sql, timeout_s=timeout_s, max_rows=MAX_ROWS + 1)
    except UnsafeSQLError as e:
        return ExecResult(error="unsafe", message=str(e))
    except QueryTimeoutError as e:
        return ExecResult(error="timeout", message=str(e))
    except sqlite3.Error as e:
        return ExecResult(error="exec_error", message=f"{type(e).__name__}: {e}")
    if len(rows) > MAX_ROWS:
        return ExecResult(error="too_many_rows", message=f"more than {MAX_ROWS} rows")
    return ExecResult(rows=[tuple(r) for r in rows])


# ---------- value and row comparison ----------

def _is_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _decimals(x: float) -> int | None:
    """Number of decimals in the shortest repr of x, or None for exponent notation / non-finite."""
    if not math.isfinite(x):
        return None
    text = repr(x)
    if "e" in text or "E" in text:
        return None
    return len(text.split(".")[1].rstrip("0")) if "." in text else 0


def values_equal(gold: object, pred: object) -> bool:
    if gold is None or pred is None:
        return gold is None and pred is None
    if _is_number(gold) and _is_number(pred):
        g, p = float(gold), float(pred)  # type: ignore[arg-type]
        if math.isclose(g, p, rel_tol=1e-6, abs_tol=1e-9):
            return True
        if isinstance(pred, float):
            d = _decimals(pred)
            if d is not None and 1 <= d <= 4:
                return math.isclose(round(g, d), p, rel_tol=0, abs_tol=10 ** -(d + 3))
        return False
    if isinstance(gold, bytes) or isinstance(pred, bytes):
        return gold == pred
    return str(gold) == str(pred)


def _canonical(v: object) -> object:
    if _is_number(v):
        f = float(v)  # type: ignore[arg-type]
        return round(f, 6) if math.isfinite(f) else f
    return v


def _rows_match(gold: list[Row], pred: list[Row], ordered: bool) -> bool:
    if len(gold) != len(pred):
        return False
    if ordered:
        return all(len(g) == len(p) and all(values_equal(a, b) for a, b in zip(g, p, strict=True))
                   for g, p in zip(gold, pred, strict=True))
    # Fast path: exact multiset equality after light canonicalisation.
    if Counter(tuple(map(_canonical, r)) for r in gold) == Counter(tuple(map(_canonical, r)) for r in pred):
        return True
    if len(gold) > MAX_ROWS_FOR_TOLERANT_MATCH:
        return False
    # Tolerant path: greedy one-to-one matching with values_equal.
    unused = list(range(len(gold)))
    for p in pred:
        for k, gi in enumerate(unused):
            g = gold[gi]
            if len(g) == len(p) and all(values_equal(a, b) for a, b in zip(g, p, strict=True)):
                del unused[k]
                break
        else:
            return False
    return True


def _project(rows: list[Row], cols: tuple[int, ...]) -> list[Row]:
    return [tuple(r[c] for c in cols) for r in rows]


def results_match(gold: list[Row], pred: list[Row], ordered: bool, allow_extra_columns: bool) -> bool:
    if not gold or not pred:
        return gold == pred
    g_cols, p_cols = len(gold[0]), len(pred[0])
    if p_cols < g_cols or (p_cols > g_cols and not allow_extra_columns):
        return False
    identity = tuple(range(g_cols))
    if p_cols == g_cols and _rows_match(gold, pred, ordered):
        return True
    if p_cols > MAX_PERMUTED_COLUMNS:
        return False          # too wide to search: only the gold column order (checked above) is accepted
    return any(_rows_match(gold, _project(pred, cols), ordered)
               for cols in itertools.permutations(range(p_cols), g_cols)
               if not (p_cols == g_cols and cols == identity))


# ---------- scoring ----------

def score(db_path: str | Path, gold_sql: str, pred_text: str, order_matters: bool = False,
          gold_rows: list[Row] | None = None, timeout_s: float = 10.0) -> Score:
    """Score one prediction. `pred_text` may be a raw model reply; the SQL is extracted from it."""
    if gold_rows is None:
        gold = run_sql(db_path, gold_sql, timeout_s=max(timeout_s, 60.0))
        if gold.rows is None:
            raise RuntimeError(f"gold SQL failed ({gold.error}): {gold.message}")
        gold_rows = gold.rows
    pred = run_sql(db_path, extract_sql(pred_text), timeout_s=timeout_s)
    if pred.rows is None:
        return Score(False, False, error=pred.error, message=pred.message)
    strict = results_match(gold_rows, pred.rows, order_matters, allow_extra_columns=False)
    relaxed = strict or results_match(gold_rows, pred.rows, order_matters, allow_extra_columns=True)
    return Score(strict, relaxed, n_pred_rows=len(pred.rows))
