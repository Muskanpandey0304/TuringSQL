from turingsql.sqlutil import normalize_whitespace, top_level_order_by


def test_top_level_order_by():
    assert top_level_order_by("SELECT a FROM t ORDER BY a")
    assert not top_level_order_by("SELECT a FROM t")
    assert not top_level_order_by("SELECT a FROM (SELECT a FROM t ORDER BY a LIMIT 3)")
    assert not top_level_order_by("SELECT a, RANK() OVER (ORDER BY b) FROM t")
    assert top_level_order_by("SELECT a FROM t UNION SELECT b FROM u ORDER BY 1")


def test_normalize_whitespace_keeps_literals():
    assert normalize_whitespace("SELECT  a\n FROM t  WHERE b = 'x  y' ;") == "SELECT a FROM t WHERE b = 'x  y'"
    assert normalize_whitespace('SELECT "a  b"   FROM t') == 'SELECT "a  b" FROM t'
