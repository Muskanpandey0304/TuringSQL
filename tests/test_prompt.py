from turingsql.prompt import SYSTEM, build_messages, extract_sql, user_content


def test_user_content_order_and_definitions():
    text = user_content("CREATE TABLE t (a INT);", "How many?", ["Revenue: sum of price."])
    assert text.index("### Schema") < text.index("### Definitions") < text.index("### Question")
    assert "Revenue: sum of price." in text


def test_user_content_without_definitions_has_no_block():
    assert "### Definitions" not in user_content("s", "q")
    assert "### Definitions" not in user_content("s", "q", [])


def test_build_messages_inference_and_training():
    infer = build_messages("s", "q")
    assert [m["role"] for m in infer] == ["system", "user"]
    assert infer[0]["content"] == SYSTEM
    train = build_messages("s", "q", sql="  SELECT 1  ")
    assert train[-1] == {"role": "assistant", "content": "SELECT 1"}


def test_extract_sql_variants():
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("SELECT a FROM t") == "SELECT a FROM t"
    assert extract_sql("SELECT 1; DROP TABLE t;") == "SELECT 1"
    assert extract_sql("```sql\nSELECT 2") == "SELECT 2"          # unclosed fence
    assert extract_sql("Here you go:\n```\nSELECT 3\n```") == "SELECT 3"
    assert extract_sql("   ") == ""
