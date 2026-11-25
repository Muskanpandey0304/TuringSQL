"""Table-level schema pruning for large databases (experiment E4, PROJECT_PLAN.md section 4.2).

The schema text (turingsql.schema) is split into one block per table. Tables are ranked by BM25 between the
question and the table block (name, columns, sample values, with snake_case/camelCase split into words), the top-k
are kept, and then tables that a kept table references by FOREIGN KEY are added back (one hop), so join paths are
not cut. The output keeps the original table order, so the prompt stays deterministic.

Before a model ever sees a pruned schema, `table_recall` measures how often every table the gold SQL uses
survives. If that number is low, pruning can only hurt.
"""
from __future__ import annotations

import re

from turingsql.fewshot import tables_in_sql

_CREATE_RE = re.compile(r"^CREATE TABLE (\S+) \(", re.MULTILINE)
_FK_RE = re.compile(r"FOREIGN KEY \([^)]*\) REFERENCES (\w+)\(")


def split_blocks(schema_text: str) -> dict[str, str]:
    """Table name (lower case) -> its block of schema text, in original order."""
    blocks: dict[str, str] = {}
    for block in schema_text.split("\n\n"):
        m = _CREATE_RE.search(block)
        if m:
            blocks[m.group(1).lower()] = block
    return blocks


def _words(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [w for w in re.findall(r"[a-z0-9]+", text.replace("_", " ").lower()) if len(w) > 1]


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def rank_tables(schema_text: str, question: str) -> list[str]:
    from rank_bm25 import BM25Okapi

    blocks = split_blocks(schema_text)
    names = list(blocks)
    if not names:
        return []
    docs = [[_stem(w) for w in _words(n + " " + n + " " + blocks[n])] for n in names]  # name counted twice
    scores = BM25Okapi(docs).get_scores([_stem(w) for w in _words(question)])
    return [names[i] for i in sorted(range(len(names)), key=lambda i: (-scores[i], i))]


def prune_schema(schema_text: str, question: str, k: int, fk_hop: bool = True) -> tuple[str, list[str]]:
    """Return (pruned schema text, kept table names). Small schemas (<= k tables) are returned unchanged."""
    blocks = split_blocks(schema_text)
    if len(blocks) <= k:
        return schema_text, list(blocks)
    kept = set(rank_tables(schema_text, question)[:k])
    if fk_hop:
        for t in list(kept):
            kept |= {r.lower() for r in _FK_RE.findall(blocks[t]) if r.lower() in blocks}
    ordered = [t for t in blocks if t in kept]
    return "\n\n".join(blocks[t] for t in ordered), ordered


def table_recall(gold_sql: str, kept_tables: list[str]) -> bool | None:
    """True if every table the gold SQL uses was kept; None if the SQL could not be parsed."""
    used = tables_in_sql(gold_sql)
    if not used:
        return None
    return used <= {t.lower() for t in kept_tables}
