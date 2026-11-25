"""Measure table-pruning recall and prompt savings on a processed dataset (no model needed).

    python scripts/pruning_recall.py --dataset spider-dev --k 3 4 5 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turingsql.pruning import prune_schema, split_blocks, table_recall  # noqa: E402

PROCESSED = ROOT / "data" / "processed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="spider-dev")
    ap.add_argument("--k", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--no-fk-hop", action="store_true")
    args = ap.parse_args()
    rows = [json.loads(x) for x in (PROCESSED / f"{args.dataset}.jsonl").read_text(encoding="utf-8").splitlines()]
    schemas = json.loads((PROCESSED / f"schemas_{args.dataset}.json").read_text(encoding="utf-8"))
    n_tables = [len(split_blocks(schemas[r["db_id"]])) for r in rows]
    print(f"{args.dataset}: {len(rows)} questions, tables per schema mean {sum(n_tables) / len(rows):.1f}, "
          f"max {max(n_tables)}")
    print("k   recall(all gold tables kept)   kept tables  schema chars kept")
    for k in args.k:
        ok = n = kept_t = 0
        chars_kept = chars_all = 0
        for r in rows:
            full = schemas[r["db_id"]]
            pruned, kept = prune_schema(full, r["question"], k, fk_hop=not args.no_fk_hop)
            rec = table_recall(r["gold_sql"], kept)
            if rec is None:
                continue
            n += 1
            ok += rec
            kept_t += len(kept)
            chars_kept += len(pruned)
            chars_all += len(full)
        print(f"{k:<3} {100 * ok / n:6.1f}% ({ok}/{n})               {kept_t / n:5.1f}        "
              f"{100 * chars_kept / chars_all:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
