"""Copy the Olist mart tables into data/olist/olist_marts.db, the database the model is evaluated on.

Only the 7 analyst-facing tables are copied; staging (stg_*) tables stay hidden, as a real analyst would
not query them. The copy is frozen so a rebuild of the Olist pipeline cannot change the benchmark.

A content hash (rows sorted, per table) is written to data/olist/marts_manifest.json. The .db file bytes
are not hashed because SQLite pages can differ while the data is identical.

Usage:
    python data_prep/build_olist_marts.py [--source PATH] [--out PATH] [--check]

--check rebuilds nothing; it recomputes the hash of the existing copy and fails if it changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parent / "HEHE" / "olist-revenue-leak-analysis" / "data" / "olist.db"
DEFAULT_OUT = ROOT / "data" / "olist" / "olist_marts.db"
MANIFEST = ROOT / "data" / "olist" / "marts_manifest.json"

MART_TABLES = (
    "dim_customer", "dim_date", "dim_product", "dim_seller",
    "fact_order_items", "fact_orders", "rfm_segments",
)


def table_hash(conn: sqlite3.Connection, table: str) -> tuple[int, str]:
    """Row count and sha256 of the table's rows in a canonical order (all columns ascending)."""
    cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
    order = ", ".join(f'"{c}"' for c in cols)
    h = hashlib.sha256()
    h.update(("|".join(cols) + "\n").encode())
    n = 0
    for row in conn.execute(f'SELECT * FROM "{table}" ORDER BY {order}'):
        h.update((repr(row) + "\n").encode())
        n += 1
    return n, h.hexdigest()


def content_manifest(db_path: Path) -> dict:
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        tables = {}
        for t in MART_TABLES:
            n, digest = table_hash(conn, t)
            tables[t] = {"rows": n, "sha256": digest}
    finally:
        conn.close()
    combined = hashlib.sha256("".join(tables[t]["sha256"] for t in MART_TABLES).encode()).hexdigest()
    return {"tables": tables, "combined_sha256": combined}


def build(source: Path, out: Path) -> None:
    if not source.is_file():
        sys.exit(f"source database not found: {source}\nRun the Olist pipeline first (pipeline/run_pipeline.py).")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)

    src = sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)
    dst = sqlite3.connect(tmp)
    try:
        for t in MART_TABLES:
            (ddl,) = src.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone()
            dst.execute(ddl)
            cols = len(src.execute(f'SELECT * FROM "{t}" LIMIT 0').description)
            rows = src.execute(f'SELECT * FROM "{t}"')
            dst.executemany(f'INSERT INTO "{t}" VALUES ({", ".join("?" * cols)})', rows)
        dst.commit()
        dst.execute("VACUUM")
    finally:
        src.close()
        dst.close()
    tmp.replace(out)

    try:
        source_label = source.resolve().relative_to(ROOT.parent.resolve()).as_posix()
    except ValueError:
        source_label = source.name
    manifest = {"source": source_label, **content_manifest(out)}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for t, info in manifest["tables"].items():
        print(f"{t:18s} {info['rows']:>8,d} rows")
    print(f"combined content sha256: {manifest['combined_sha256']}")
    print(f"wrote {out} and {MANIFEST}")


def check(out: Path) -> int:
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    actual = content_manifest(out)
    if actual["combined_sha256"] == expected["combined_sha256"]:
        print("olist_marts.db matches the manifest")
        return 0
    for t in MART_TABLES:
        if actual["tables"][t] != expected["tables"][t]:
            print(f"CHANGED: {t}: {expected['tables'][t]} -> {actual['tables'][t]}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        return check(args.out)
    build(args.source, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
