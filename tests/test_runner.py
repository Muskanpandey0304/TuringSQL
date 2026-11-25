import json

from turingsql.eval.runner import Generation, RunConfig, run


def test_run_writes_scored_report(tiny_db, tmp_path):
    examples = [
        {"id": "a", "level": "easy", "db_path": tiny_db, "question": "late?", "order_matters": False,
         "gold_sql": "SELECT COUNT(*) FROM fact_orders WHERE is_late = 1"},
        {"id": "b", "level": "hard", "db_path": tiny_db, "question": "all?", "order_matters": False,
         "gold_sql": "SELECT COUNT(*) FROM fact_orders"},
        {"id": "c", "level": "hard", "db_path": tiny_db, "question": "boom", "order_matters": False,
         "gold_sql": "SELECT 1"},
    ]
    seen = []

    def fake(messages):
        seen.append(messages)
        q = messages[-1]["content"]
        if "boom" in q:
            raise TimeoutError("server gone")
        sql = "SELECT SUM(is_late) FROM fact_orders" if "late?" in q else "SELECT 99"
        return Generation(f"```sql\n{sql}\n```", 0.5, prompt_tokens=100, completion_tokens=10, cached_tokens=40)

    cfg = RunConfig(run_id="t", model="fake", engine="test", dataset="tiny", concurrency=2)
    summary = run(examples, fake, cfg, tmp_path, progress=False)

    assert summary["ex_strict"]["n"] == 3
    assert summary["ex_strict"]["acc"] == round(1 / 3, 4)
    assert summary["errors"] == {"generation_error": 1}
    assert summary["by_level"]["easy"]["acc"] == 1.0
    assert summary["cached_token_share"] == 0.4
    assert "CREATE TABLE fact_orders" in seen[0][1]["content"]
    recs = [json.loads(x) for x in (tmp_path / "t" / "predictions.jsonl").read_text().splitlines()]
    assert [r["id"] for r in recs] == ["a", "b", "c"]          # input order kept despite concurrency
    assert recs[0]["pred_sql"] == "SELECT SUM(is_late) FROM fact_orders"
    assert (tmp_path / "t" / "summary.md").read_text().startswith("# Run `t`")
