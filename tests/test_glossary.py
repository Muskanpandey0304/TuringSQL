import json
from pathlib import Path

from turingsql.retrieval.glossary import fuse_rrf, term_in_query

GLOSSARY = Path(__file__).resolve().parents[1] / "data" / "olist" / "glossary.jsonl"


def test_rrf_rewards_agreement():
    scores = fuse_rrf([[1, 2, 3], [2, 1, 4]])
    assert scores[1] == scores[2] > scores[3] == scores[4]
    assert fuse_rrf([[7], []]) == {7: 1 / 61}


def test_term_in_query():
    assert term_in_query("Revenue", "How much revenue did each region bring in?")
    assert term_in_query("Average order value (AOV)", "What is our AOV by state?")
    assert term_in_query("Order", "How many orders were placed?")          # plural
    assert not term_in_query("Late rate", "How many late deliveries were there?")
    assert not term_in_query("Revenue", "What is the revenues_total column?")   # whole words only


def test_glossary_entries_well_formed():
    entries = [json.loads(x) for x in GLOSSARY.read_text(encoding="utf-8").splitlines()]
    assert len({e["id"] for e in entries}) == len(entries)
    for e in entries:
        assert e["term"] and e["text"] and e["plain"].startswith(e["term"] + ":")
