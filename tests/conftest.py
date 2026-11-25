import sqlite3

import pytest


@pytest.fixture
def tiny_db(tmp_path):
    """A small two-table SQLite file shaped like the Olist marts."""
    path = tmp_path / "tiny.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE dim_customer (customer_unique_id TEXT PRIMARY KEY, state TEXT);
        CREATE TABLE fact_orders (
            order_id TEXT PRIMARY KEY,
            customer_unique_id TEXT REFERENCES dim_customer(customer_unique_id),
            is_late INTEGER, review_score INTEGER, note TEXT
        );
        INSERT INTO dim_customer VALUES ('c1', 'SP'), ('c2', 'RJ');
        INSERT INTO fact_orders VALUES
            ('o1', 'c1', 0, 5, 'fine'),
            ('o2', 'c1', 1, 2, 'a very long note that goes on and on and on and on and on'),
            ('o3', 'c2', 1, 1, NULL);
        """
    )
    conn.commit()
    conn.close()
    return path
