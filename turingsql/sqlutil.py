"""Small SQL helpers shared by data preparation and evaluation."""
from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

_ORDER_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)


def top_level_order_by(sql: str) -> bool:
    """True if the outermost query sorts its result (ORDER BY on the final SELECT / set operation).

    An ORDER BY inside a subquery or window does not make the *result* order meaningful. Falls back to a
    plain text search when sqlglot cannot parse the query.
    """
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.ParseError:
        return bool(_ORDER_RE.search(sql))
    return isinstance(tree, exp.Query) and tree.args.get("order") is not None


def normalize_whitespace(sql: str) -> str:
    """Collapse runs of whitespace outside string literals; strip a trailing semicolon."""
    out, quote = [], None
    prev_space = False
    for ch in sql.strip().rstrip(";").strip():
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            prev_space = False
        elif ch.isspace():
            if not prev_space:
                out.append(" ")
            prev_space = True
        else:
            out.append(ch)
            prev_space = False
    return "".join(out).strip()
