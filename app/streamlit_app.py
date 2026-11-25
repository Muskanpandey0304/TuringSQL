"""Ask Olist: plain-English questions about the Olist marketplace, answered with SQL you can see and edit.

    # 1. serve a model (base or fine-tuned) with TuringServe or any OpenAI-compatible server
    turingserve serve --model models/Qwen2.5-Coder-1.5B-Instruct --served-model-name qwen-coder-1.5b \
        --device cpu --kernels reference --dtype float32 --port 8011
    # 2. run the app
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turingsql.ask import AskOlist  # noqa: E402
from turingsql.benchmark import OLIST_DIR  # noqa: E402

EXAMPLES = [
    "What share of customers bought more than once?",
    "Which 5 states have the highest late-delivery rate for delivered orders?",
    "What is the average review score for late orders compared with on-time orders?",
    "Monthly revenue in 2018",
    "Which 10 product categories bring in the most revenue?",
    "How long does delivery take on average in each region?",
]

st.set_page_config(page_title="Ask Olist", layout="wide")


@st.cache_resource(show_spinner="Loading glossary retrieval (first run downloads small models)...")
def load_retriever():
    from turingsql.retrieval.glossary import GlossaryRetriever

    return GlossaryRetriever.from_jsonl(OLIST_DIR / "glossary.jsonl")


def make_generate(base_url: str, model: str, max_tokens: int = 256):
    from openai import OpenAI

    url = base_url.rstrip("/")
    client = OpenAI(base_url=url if url.endswith("/v1") else url + "/v1", api_key="none", timeout=600)

    def generate(messages):
        r = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens,
                                           temperature=0, top_p=1)
        return r.choices[0].message.content or ""

    return generate


with st.sidebar:
    st.header("Model server")
    base_url = st.text_input("OpenAI-compatible URL", "http://127.0.0.1:8011")
    model = st.text_input("Served model name", "qwen-coder-1.5b")
    use_glossary = st.toggle("Add business definitions (glossary retrieval)", value=True)
    st.caption("SQL runs read-only on a frozen copy of the Olist marts, one SELECT at a time, "
               "with a 10 s timeout and LIMIT 1000.")

st.title("Ask Olist")
st.write("Ask a question about 100,000 Brazilian marketplace orders (2016-2018). "
         "The model writes SQL; you see the SQL, the definitions it used, and the result.")

db = OLIST_DIR / "olist_marts.db"
if not db.is_file():
    st.error("data/olist/olist_marts.db not found: run `python data_prep/build_olist_marts.py` first.")
    st.stop()

cols = st.columns(3)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 3].button(ex, use_container_width=True):
        st.session_state["question"] = ex

question = st.text_input("Your question", key="question")

if question:
    retrieve = None
    if use_glossary:
        r = load_retriever()

        def retrieve(q: str) -> list[str]:
            return [h.text for h in r.search(q, k=3, min_prob=0.1)]

    app = AskOlist(db, make_generate(base_url, model), retrieve=retrieve)
    if st.session_state.get("last_q") != question:
        with st.spinner("Writing SQL..."):
            st.session_state["answer"] = app.ask(question)
        st.session_state["last_q"] = question
    ans = st.session_state["answer"]

    left, right = st.columns([2, 3])
    with left:
        st.subheader("SQL")
        edited = st.text_area("Generated SQL (edit and re-run if needed)", ans.sql, height=200)
        if st.button("Re-run SQL") and edited.strip():
            ans = app.ask(question, sql_override=edited)
            ans.definitions = st.session_state["answer"].definitions
            st.session_state["answer"] = ans
        if ans.definitions:
            with st.expander(f"Business definitions used ({len(ans.definitions)})"):
                for d in ans.definitions:
                    st.markdown(f"- {d}")
        t = ans.timings
        st.caption(" · ".join(f"{k.replace('_s', '')} {v:.1f}s" for k, v in t.items()))
    with right:
        st.subheader("Result")
        if ans.error:
            st.error(ans.error)
        else:
            df = pd.DataFrame(ans.rows, columns=ans.columns)
            if ans.chart:
                data = df.set_index(ans.chart["x"])[ans.chart["y"]]
                (st.line_chart if ans.chart["type"] == "line" else st.bar_chart)(data)
            st.dataframe(df, use_container_width=True)
            if ans.limited and len(ans.rows) >= app.limit:
                st.caption(f"Showing the first {app.limit} rows.")
