"""Prepare Spider and BIRD for training and evaluation (PROJECT_PLAN.md section 3.2).

For each dataset this writes
    data/processed/<name>.jsonl              one example per line: id, db_id, db_path, question, gold_sql,
                                             order_matters, evidence (BIRD), source
    data/processed/schemas_<name>.json       db_id -> schema text (the exact text the model sees)
    data/processed/prep_report_<name>.json   counts of everything dropped and why

Cleaning rules: drop examples whose gold SQL fails our safety check, errors, times out (10 s) or returns
more than 10,000 rows; drop exact duplicate (db_id, question) pairs; normalise whitespace in the SQL.

Sources (all CC BY-SA 4.0):
    spider:      spider_data.zip (official zip; train_spider.json + train_others.json, dev.json, database/)
    bird-dev:    dev.zip from bird-bench (dev.json + nested dev_databases.zip)
    bird-train:  train.zip from bird-bench (train.json + nested train_databases.zip). The databases are
                 32 GB unzipped, so they are extracted ONE AT A TIME into a temp dir, processed and deleted.
                 Run this part where there is fast network and ~15 GB of free disk (e.g. Kaggle).

Usage:
    python data_prep/prepare_public.py spider   --zip data/raw/spider/spider_data.zip
    python data_prep/prepare_public.py bird-dev  --zip data/raw/bird/dev.zip
    python data_prep/prepare_public.py bird-train --zip /kaggle/tmp/train.zip --workdir /kaggle/tmp/bird
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turingsql.eval.execution import MAX_ROWS, run_sql  # noqa: E402
from turingsql.schema import serialize_schema  # noqa: E402
from turingsql.sqlutil import normalize_whitespace, top_level_order_by  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
SCHEMA_ARGS = {"n_sample_rows": 2, "max_value_len": 24}   # must match the evaluation prompt


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def _find(z: zipfile.ZipFile, suffix: str) -> str:
    hits = [n for n in z.namelist() if n.endswith(suffix) and "__MACOSX" not in n]
    if not hits:
        raise FileNotFoundError(f"{suffix} not found in {z.filename}")
    return min(hits, key=len)


def _safe_extract(z: zipfile.ZipFile, members: list[str], dest: Path) -> None:
    dest = dest.resolve()
    for m in members:
        target = (dest / m).resolve()
        if dest not in target.parents and target != dest:
            raise ValueError(f"unsafe path in zip: {m}")
    z.extractall(dest, members)


# ---------- core: validate examples against one database ----------

def process_db(db_id: str, db_path: Path, items: list[dict], stats: Counter, keep_db_path: bool,
               source: str) -> tuple[list[dict], str]:
    schema = serialize_schema(db_path, **SCHEMA_ARGS)
    kept = []
    for it in items:
        sql = normalize_whitespace(it["sql"])
        res = run_sql(db_path, sql, timeout_s=10)
        if res.rows is None:
            stats[f"drop_{res.error}"] += 1
            continue
        if not res.rows:
            stats["kept_empty_result"] += 1      # kept: an empty answer is still checkable
        kept.append({
            "id": it["id"], "db_id": db_id,
            "db_path": _rel(db_path) if keep_db_path else None,
            "question": it["question"].strip(), "gold_sql": sql,
            "order_matters": top_level_order_by(sql),
            "evidence": (it.get("evidence") or "").strip() or None,
            "source": source,
        })
    stats["kept"] += len(kept)
    return kept, schema


def group_items(raw: list[dict], prefix: str, sql_key: str, stats: Counter) -> dict[str, list[dict]]:
    by_db: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for i, r in enumerate(raw):
        key = (r["db_id"], r["question"].strip().lower())
        stats["raw"] += 1
        if key in seen:
            stats["drop_duplicate"] += 1
            continue
        seen.add(key)
        by_db[r["db_id"]].append({"id": f"{prefix}-{i:05d}", "question": r["question"], "sql": r[sql_key],
                                  "evidence": r.get("evidence")})
    return by_db


def write_outputs(name: str, examples: list[dict], schemas: dict[str, str], stats: Counter, extra: dict) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with (PROCESSED / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    (PROCESSED / f"schemas_{name}.json").write_text(json.dumps(schemas, ensure_ascii=False), encoding="utf-8")
    report = {"dataset": name, "schema_args": SCHEMA_ARGS, "max_rows": MAX_ROWS, "databases": len(schemas),
              **dict(sorted(stats.items())), **extra}
    (PROCESSED / f"prep_report_{name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


# ---------- Spider ----------

def prepare_spider(zip_path: Path) -> None:
    dest = ROOT / "data" / "spider"
    with zipfile.ZipFile(zip_path) as z:
        train = json.loads(z.read(_find(z, "/train_spider.json")))
        others = json.loads(z.read(_find(z, "/train_others.json")))
        dev = json.loads(z.read(_find(z, "/dev.json")))
        db_members = [n for n in z.namelist() if "/database/" in n and n.endswith(".sqlite")
                      and "__MACOSX" not in n and "/test_database/" not in n]
        if not (dest / "database").is_dir():
            _safe_extract(z, db_members, dest / "_extract")
            root = Path(db_members[0]).parts[0]
            shutil.move(str(dest / "_extract" / root / "database"), str(dest / "database"))
            shutil.rmtree(dest / "_extract")
    for name, raw in (("spider-train", train + others), ("spider-dev", dev)):
        stats: Counter = Counter()
        by_db = group_items(raw, name, "query", stats)
        examples, schemas = [], {}
        for db_id, items in sorted(by_db.items()):
            db_path = dest / "database" / db_id / f"{db_id}.sqlite"
            if not db_path.is_file():
                stats["drop_missing_db"] += len(items)
                continue
            kept, schemas[db_id] = process_db(db_id, db_path, items, stats, True, "spider")
            examples += kept
        write_outputs(name, examples, schemas, stats, {"source_zip": zip_path.name})


# ---------- BIRD ----------

def iter_nested_dbs(inner_zip: Path, workdir: Path) -> Iterator[tuple[str, Path]]:
    """Extract one database folder at a time from BIRD's nested *_databases.zip, then delete it."""
    with zipfile.ZipFile(inner_zip) as z:
        by_db: dict[str, list[str]] = defaultdict(list)
        for n in z.namelist():
            parts = Path(n).parts
            if "__MACOSX" in parts or len(parts) < 3 or n.endswith("/"):
                continue
            by_db[parts[-2]].append(n)
        for db_id, members in sorted(by_db.items()):
            sqlite = [m for m in members if m.endswith(".sqlite")]
            if not sqlite:
                continue
            tmp = Path(tempfile.mkdtemp(dir=workdir))
            try:
                _safe_extract(z, sqlite, tmp)
                yield db_id, tmp / sqlite[0]
            finally:
                shutil.rmtree(tmp, ignore_errors=True)


def prepare_bird(split: str, zip_path: Path, workdir: Path | None) -> None:
    name = f"bird-{split}"
    keep = split == "dev"          # dev databases stay on disk (needed for evaluation); train ones are deleted
    work = workdir or (ROOT / "data" / "bird")
    work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        raw = json.loads(z.read(_find(z, f"/{split}.json")))
        inner_name = _find(z, f"{split}_databases.zip")
        inner = work / Path(inner_name).name
        if not inner.is_file():
            _safe_extract(z, [inner_name], work / "_outer")
            shutil.move(str(work / "_outer" / inner_name), str(inner))
            shutil.rmtree(work / "_outer")
    stats: Counter = Counter()
    by_db = group_items(raw, name, "SQL", stats)
    examples, schemas = [], {}
    db_root = ROOT / "data" / "bird" / f"{split}_databases"
    for db_id, tmp_path in iter_nested_dbs(inner, work):
        if db_id not in by_db:
            continue
        path = tmp_path
        if keep:
            path = db_root / db_id / f"{db_id}.sqlite"
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_path, path)
        kept, schemas[db_id] = process_db(db_id, path, by_db.pop(db_id), stats, keep, "bird")
        examples += kept
        print(f"  {db_id}: kept {len(kept)}", flush=True)
    for items in by_db.values():
        stats["drop_missing_db"] += len(items)
    write_outputs(name, examples, schemas, stats, {"source_zip": zip_path.name})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["spider", "bird-dev", "bird-train"])
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, default=None, help="scratch space for BIRD's nested zips")
    args = ap.parse_args()
    if args.dataset == "spider":
        prepare_spider(args.zip)
    else:
        prepare_bird(args.dataset.split("-")[1], args.zip, args.workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
