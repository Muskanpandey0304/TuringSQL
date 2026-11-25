"""Export real TuringSQL prompts as a TuringServe benchmark workload (PROJECT_PLAN.md section 7.2).

TuringServe's `rag` workload reads a JSONL of {"system", "prompt"} from $TS_RAG_PROMPTS, so no change to
TuringServe is needed:

    python scripts/export_bench_prompts.py --out benchmarks/turingsql_prompts.jsonl
    # in the TuringServe repo:
    TS_RAG_PROMPTS=/path/to/turingsql_prompts.jsonl python -m benchmarks.serve_bench --dataset rag ...

The mix: every Olist dev question (all share one schema, so this is the prefix-cache-friendly case) plus a
fixed sample of Spider dev questions over 20 different schemas (the cache-unfriendly case). The Olist test
split is never exported.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turingsql.benchmark import OLIST_DIR, load_questions  # noqa: E402
from turingsql.prompt import SYSTEM, user_content  # noqa: E402
from turingsql.schema import serialize_schema  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "turingsql_prompts.jsonl")
    ap.add_argument("--spider", type=int, default=160, help="Spider dev questions to add (0 for Olist only)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    olist_schema = serialize_schema(OLIST_DIR / "olist_marts.db")
    for q in load_questions("dev"):
        rows.append({"system": SYSTEM, "prompt": user_content(olist_schema, q["question"]),
                     "source": "olist", "id": q["id"]})
    spider_path = ROOT / "data" / "processed" / "spider-dev.jsonl"
    if args.spider and spider_path.is_file():
        schemas = json.loads((ROOT / "data" / "processed" / "schemas_spider-dev.json").read_text(encoding="utf-8"))
        spider = [json.loads(x) for x in spider_path.read_text(encoding="utf-8").splitlines()]
        for ex in random.Random(args.seed).sample(spider, min(args.spider, len(spider))):
            rows.append({"system": SYSTEM, "prompt": user_content(schemas[ex["db_id"]], ex["question"]),
                         "source": "spider", "id": ex["id"]})
    random.Random(args.seed).shuffle(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} prompts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
