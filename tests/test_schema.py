import pytest

from turingsql.schema import connect_readonly, serialize_schema


def test_tables_sorted_and_ddl_normalised(tiny_db):
    text = serialize_schema(tiny_db)
    assert text.index("CREATE TABLE dim_customer") < text.index("CREATE TABLE fact_orders")
    assert "customer_unique_id TEXT PRIMARY KEY" in text
    assert "FOREIGN KEY (customer_unique_id) REFERENCES dim_customer(customer_unique_id)" in text


def test_sample_rows_truncated_and_null(tiny_db):
    text = serialize_schema(tiny_db, n_sample_rows=3, max_value_len=20)
    assert "NULL" in text
    assert "a very long note..." in text
    assert "on and on and on and on" not in text


def test_deterministic(tiny_db):
    assert serialize_schema(tiny_db) == serialize_schema(tiny_db)


def test_subset_and_missing_table(tiny_db):
    text = serialize_schema(tiny_db, tables=["fact_orders"], n_sample_rows=0)
    assert "dim_customer (" not in text and "sample rows" not in text
    with pytest.raises(KeyError):
        serialize_schema(tiny_db, tables=["nope"])


def test_readonly_connection_cannot_write(tiny_db):
    import sqlite3

    conn = connect_readonly(tiny_db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM fact_orders")
    conn.close()


def test_missing_file_is_not_created(tmp_path):
    with pytest.raises(FileNotFoundError):
        connect_readonly(tmp_path / "missing.db")
    assert not (tmp_path / "missing.db").exists()
