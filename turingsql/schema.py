"""Turn a SQLite database into the schema text the model sees.

The output is deterministic: tables in alphabetical order, columns in database order, sample rows
taken by rowid. A stable schema text means a stable prompt prefix, which TuringServe's prefix cache
can reuse across questions (PROJECT_PLAN.md section 7.3).
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite file read-only. Fails if the file does not exist (never creates one)."""
    path = Path(db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)


def list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _fmt(value: object, max_len: int) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        text = f"{value:.6g}"           # -22.89456108964741 -> -22.8946; keeps prompts short
    else:
        text = str(value).replace("\n", " ")
    return text if len(text) <= max_len else text[: max_len - 3].rstrip() + "..."


def table_ddl(conn: sqlite3.Connection, table: str) -> str:
    """A normalised CREATE TABLE line built from PRAGMA info, not the stored (free-form) SQL."""
    cols = conn.execute(f"PRAGMA table_info({_quote(table)})").fetchall()
    # table_info rows: (cid, name, type, notnull, default, pk)
    col_defs = []
    for _, name, ctype, _, _, pk in cols:
        col = f"{name} {ctype or 'TEXT'}".strip()
        if pk:
            col += " PRIMARY KEY"
        col_defs.append(col)
    fks = conn.execute(f"PRAGMA foreign_key_list({_quote(table)})").fetchall()
    # foreign_key_list rows: (id, seq, table, from, to, on_update, on_delete, match)
    for fk in fks:
        col_defs.append(f"FOREIGN KEY ({fk[3]}) REFERENCES {fk[2]}({fk[4]})")
    return f"CREATE TABLE {table} ({', '.join(col_defs)});"


def sample_rows(conn: sqlite3.Connection, table: str, n: int, max_value_len: int) -> list[str]:
    if n <= 0:
        return []
    rows = conn.execute(f"SELECT * FROM {_quote(table)} LIMIT {int(n)}").fetchall()
    return ["-- " + " | ".join(_fmt(v, max_value_len) for v in row) for row in rows]


def serialize_schema(db_path: str | Path, tables: Iterable[str] | None = None, n_sample_rows: int = 2,
                     max_value_len: int = 24) -> str:
    """Schema text for the prompt. `tables` limits it to a subset (schema pruning); order is still sorted."""
    with connect_readonly(db_path) as conn:
        available = list_tables(conn)
        chosen = available if tables is None else sorted(set(tables))
        missing = set(chosen) - set(available)
        if missing:
            raise KeyError(f"tables not in database: {sorted(missing)}")
        blocks = []
        for t in chosen:
            lines = [table_ddl(conn, t)]
            samples = sample_rows(conn, t, n_sample_rows, max_value_len)
            if samples:
                lines.append("-- sample rows:")
                lines += samples
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
