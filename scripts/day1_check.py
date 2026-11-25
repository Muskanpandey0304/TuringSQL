"""Day-1 check: does TuringServe serve Qwen2.5-Coder and match Hugging Face greedy output?

Usage:
    python scripts/day1_check.py --model models/Qwen2.5-Coder-0.5B-Instruct --base-url http://127.0.0.1:8011
"""
from __future__ import annotations

import argparse
import time

import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from turingsql.prompt import build_messages

SCHEMA = """CREATE TABLE fact_orders (order_id TEXT, customer_unique_id TEXT, customer_state TEXT,
  order_status TEXT, purchase_date TEXT, is_delivered INTEGER, is_late INTEGER,
  delivery_days REAL, review_score INTEGER, items_price REAL, freight_total REAL);
CREATE TABLE dim_customer (customer_unique_id TEXT, city TEXT, state TEXT, region TEXT);"""

QUESTIONS = [
    "How many orders were delivered late?",
    "What is the average review score for late orders compared with on-time orders?",
    "Which 5 customer states have the most delivered orders?",
]


def messages(q: str) -> list[dict]:
    return build_messages(SCHEMA, q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8011")
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    client = OpenAI(base_url=f"{args.base_url}/v1", api_key="none")
    served = client.models.list().data[0].id

    tok = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()

    matches = 0
    for q in QUESTIONS:
        t0 = time.perf_counter()
        r = client.chat.completions.create(
            model=served, messages=messages(q), max_tokens=args.max_tokens, temperature=0
        )
        ts_text = r.choices[0].message.content or ""
        ts_time = time.perf_counter() - t0

        ids = tok.apply_chat_template(messages(q), add_generation_prompt=True, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            # Qwen's generation_config.json sets repetition_penalty=1.05, which HF applies even
            # with do_sample=False. Force pure greedy so both engines run the same algorithm.
            out = hf.generate(ids, attention_mask=torch.ones_like(ids),
                              max_new_tokens=args.max_tokens, do_sample=False,
                              repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)
        hf_text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        hf_time = time.perf_counter() - t0

        same = ts_text.strip() == hf_text.strip()
        matches += same
        print(f"\nQ: {q}")
        print(f"TuringServe ({ts_time:.1f}s): {ts_text.strip()}")
        print(f"HF generate ({hf_time:.1f}s): {hf_text.strip()}")
        print(f"identical: {same}")

    print(f"\n{matches}/{len(QUESTIONS)} outputs identical")


if __name__ == "__main__":
    main()
