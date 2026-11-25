"""Headless checks of the Streamlit page (no model server needed)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
DB = ROOT / "data" / "olist" / "olist_marts.db"

st_testing = pytest.importorskip("streamlit.testing.v1")
pytestmark = pytest.mark.skipif(not DB.is_file(), reason="needs data/olist/olist_marts.db")


def test_page_renders_without_question():
    at = st_testing.AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Ask Olist"
    assert len(at.button) >= 6


def test_unreachable_server_shows_error_not_crash():
    at = st_testing.AppTest.from_file(str(APP), default_timeout=60).run()
    at.sidebar.text_input[0].set_value("http://127.0.0.1:9")      # nothing listens here
    at.sidebar.toggle[0].set_value(False)                          # skip glossary model loading
    at.text_input(key="question").set_value("How many orders are there?").run()
    assert not at.exception
    assert any("model request failed" in e.value for e in at.error)
