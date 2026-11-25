"""Few-shot example selection for experiment E1 (PROJECT_PLAN.md section 6).

Examples come from Spider *train* only, picked by BM25 similarity of the question, at most one per database so
the shots show different schemas. Each shot is a prior user/assistant turn with a compact schema: only the
CREATE TABLE lines of the tables its SQL uses (no sample rows), to keep the prompt short.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import sqlglot
from sqlglot import exp

from turingsql.prompt import SYSTEM, user_content

_TOKEN_RE = re.compile(r"\w+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def tables_in_sql(sql: str) -> set[str]:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.ParseError:
        return set()
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    return {t.name.lower() for t in tree.find_all(exp.Table)} - ctes


def compact_schema(schema_text: str, tables: set[str]) -> str:
    """Keep only the CREATE TABLE lines (no sample rows) of the given tables."""
    keep = []
    for line in schema_text.splitlines():
        m = re.match(r"CREATE TABLE (\S+) \(", line)
        if m and m.group(1).lower() in tables:
            keep.append(line)
    return "\n".join(keep)


class FewShotSelector:
    def __init__(self, examples: list[dict], schemas: dict[str, str]):
        from rank_bm25 import BM25Okapi

        self.examples = examples
        self.schemas = schemas
        self.bm25 = BM25Okapi([_tokens(e["question"]) for e in examples])

    @classmethod
    def from_processed(cls, processed_dir: Path, name: str = "spider-train") -> FewShotSelector:
        rows = [json.loads(x) for x in (processed_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        schemas = json.loads((processed_dir / f"schemas_{name}.json").read_text(encoding="utf-8"))
        return cls(rows, schemas)

    def select(self, question: str, k: int = 3, exclude_db: str | None = None) -> list[dict]:
        scores = self.bm25.get_scores(_tokens(question))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        picked, dbs = [], set()
        for i in order:
            ex = self.examples[i]
            if ex["db_id"] in dbs or ex["db_id"] == exclude_db:
                continue
            schema = compact_schema(self.schemas[ex["db_id"]], tables_in_sql(ex["gold_sql"]))
            if not schema:
                continue
            picked.append({"schema": schema, "question": ex["question"], "sql": ex["gold_sql"], "id": ex["id"]})
            dbs.add(ex["db_id"])
            if len(picked) == k:
                break
        return picked


def build_fewshot_messages(shots: list[dict], schema: str, question: str,
                           definitions: list[str] | None = None) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for s in shots:
        msgs.append({"role": "user", "content": user_content(s["schema"], s["question"])})
        msgs.append({"role": "assistant", "content": s["sql"]})
    msgs.append({"role": "user", "content": user_content(schema, question, definitions)})
    return msgs
