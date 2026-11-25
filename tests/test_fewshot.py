from turingsql.fewshot import FewShotSelector, build_fewshot_messages, compact_schema, tables_in_sql

SCHEMA = ("CREATE TABLE singer (id INT, name TEXT);\n-- sample rows:\n-- 1 | Ann\n\n"
          "CREATE TABLE concert (id INT, singer_id INT);\n-- sample rows:\n-- 1 | 1")


def test_tables_in_sql_ignores_ctes():
    sql = "WITH x AS (SELECT * FROM singer) SELECT * FROM x JOIN concert c ON 1"
    assert tables_in_sql(sql) == {"singer", "concert"}


def test_compact_schema_keeps_only_used_tables_without_samples():
    assert compact_schema(SCHEMA, {"singer"}) == "CREATE TABLE singer (id INT, name TEXT);"


def test_selector_one_shot_per_database_and_exclusion():
    rows = [
        {"id": "a", "db_id": "music", "question": "How many singers are there?",
         "gold_sql": "SELECT COUNT(*) FROM singer"},
        {"id": "b", "db_id": "music", "question": "How many singers sang?", "gold_sql": "SELECT COUNT(*) FROM concert"},
        {"id": "c", "db_id": "other", "question": "How many singers exist?", "gold_sql": "SELECT COUNT(*) FROM singer"},
        {"id": "d", "db_id": "far", "question": "List cities", "gold_sql": "SELECT name FROM singer"},
    ]
    sel = FewShotSelector(rows, {"music": SCHEMA, "other": SCHEMA, "far": SCHEMA})
    shots = sel.select("How many singers are there?", k=2)
    assert [s["id"] for s in shots] == ["a", "c"]
    assert [s["id"] for s in sel.select("How many singers are there?", k=1, exclude_db="music")] == ["c"]


def test_fewshot_messages_shape():
    shots = [{"schema": "CREATE TABLE t (a INT);", "question": "q1", "sql": "SELECT a FROM t"}]
    msgs = build_fewshot_messages(shots, "CREATE TABLE u (b INT);", "q2")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[2]["content"] == "SELECT a FROM t" and "q2" in msgs[3]["content"]
