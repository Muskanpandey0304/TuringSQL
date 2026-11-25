import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_prep"))

import prepare_public as pp  # noqa: E402


def test_group_items_drops_duplicates():
    stats = Counter()
    raw = [{"db_id": "a", "question": "How many?", "query": "SELECT 1"},
           {"db_id": "a", "question": "how many? ", "query": "SELECT 2"},
           {"db_id": "b", "question": "How many?", "query": "SELECT 3"}]
    by_db = pp.group_items(raw, "x", "query", stats)
    assert stats["drop_duplicate"] == 1 and stats["raw"] == 3
    assert [i["sql"] for i in by_db["a"]] == ["SELECT 1"] and len(by_db["b"]) == 1


def test_process_db_filters_and_labels(tiny_db):
    stats = Counter()
    items = [
        {"id": "1", "question": "late?", "sql": "SELECT COUNT(*)  FROM fact_orders WHERE is_late = 1 ;"},
        {"id": "2", "question": "bad", "sql": "SELECT nope FROM fact_orders"},
        {"id": "3", "question": "sorted", "sql": "SELECT order_id FROM fact_orders ORDER BY order_id"},
        {"id": "4", "question": "none", "sql": "SELECT order_id FROM fact_orders WHERE 0"},
        {"id": "5", "question": "write", "sql": "DELETE FROM fact_orders"},
    ]
    kept, schema = pp.process_db("tiny", tiny_db, items, stats, keep_db_path=False, source="t")
    assert [k["id"] for k in kept] == ["1", "3", "4"]
    assert kept[0]["gold_sql"] == "SELECT COUNT(*) FROM fact_orders WHERE is_late = 1"
    assert kept[1]["order_matters"] and not kept[0]["order_matters"]
    assert stats["drop_exec_error"] == 1 and stats["drop_unsafe"] == 1 and stats["kept_empty_result"] == 1
    assert "CREATE TABLE fact_orders" in schema and kept[0]["db_path"] is None


def test_iter_nested_dbs_extracts_one_at_a_time(tiny_db, tmp_path):
    inner = tmp_path / "train_databases.zip"
    with zipfile.ZipFile(inner, "w") as z:
        for name in ("alpha", "beta"):
            z.write(tiny_db, f"train_databases/{name}/{name}.sqlite")
            z.writestr(f"train_databases/{name}/database_description/x.csv", "a,b")
        z.writestr("__MACOSX/train_databases/alpha/._alpha.sqlite", "junk")
    work = tmp_path / "work"
    work.mkdir()
    seen = []
    for db_id, path in pp.iter_nested_dbs(inner, work):
        assert path.is_file()
        assert len(list(work.iterdir())) == 1          # only the current database is on disk
        seen.append(db_id)
    assert seen == ["alpha", "beta"]
    assert list(work.iterdir()) == []                  # everything cleaned up
