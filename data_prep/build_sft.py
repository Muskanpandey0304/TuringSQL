"""Build the fine-tuning files from processed Spider / BIRD training data.

Writes data/processed/sft_train.jsonl and sft_val.jsonl: {"id", "db_id", "source", "messages": [...]} with the
schema already in the prompt (turingsql.prompt, the same template as evaluation), so training needs no
databases. Validation = all examples of a fixed 5% of training *databases* (never rows of a database that is
also in training), so validation loss measures generalisation to unseen schemas.

Examples longer than --max-len tokens are dropped and counted (never truncated).

    python data_prep/build_sft.py --sources spider-train bird-train --tokenizer models/Qwen2.5-Coder-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turingsql.prompt import PROMPT_VERSION, build_messages  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
VAL_DB_FRACTION = 0.05
SEED = 20261124


def load(name: str) -> tuple[list[dict], dict[str, str]]:
    rows = [json.loads(x) for x in (PROCESSED / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    schemas = json.loads((PROCESSED / f"schemas_{name}.json").read_text(encoding="utf-8"))
    return rows, schemas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["spider-train", "bird-train"])
    ap.add_argument("--tokenizer", default=None, help="if given, drop examples longer than --max-len tokens")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--no-evidence", action="store_true", help="leave BIRD evidence out of the prompt")
    args = ap.parse_args()

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    records: list[dict] = []
    stats: Counter = Counter()
    for name in args.sources:
        path = PROCESSED / f"{name}.jsonl"
        if not path.is_file():
            print(f"skipping {name}: {path} not found")
            stats[f"missing_source_{name}"] += 1
            continue
        rows, schemas = load(name)
        for r in rows:
            defs = [r["evidence"]] if r.get("evidence") and not args.no_evidence else None
            msgs = build_messages(schemas[r["db_id"]], r["question"], defs, sql=r["gold_sql"])
            if tok is not None:
                n = len(tok.apply_chat_template(msgs, tokenize=True))
                if n > args.max_len:
                    stats[f"drop_too_long_{name}"] += 1
                    continue
                stats["tokens"] += n
            records.append({"id": r["id"], "db_id": f"{r['source']}:{r['db_id']}", "source": r["source"],
                            "messages": msgs})
            stats[f"kept_{name}"] += 1

    dbs = sorted({r["db_id"] for r in records})
    rng = random.Random(SEED)
    val_dbs = set(rng.sample(dbs, max(1, round(VAL_DB_FRACTION * len(dbs)))))
    train = [r for r in records if r["db_id"] not in val_dbs]
    val = [r for r in records if r["db_id"] in val_dbs]
    rng.shuffle(train)

    for fname, part in (("sft_train.jsonl", train), ("sft_val.jsonl", val)):
        with (PROCESSED / fname).open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"prompt_version": PROMPT_VERSION, "sources": args.sources, "max_len": args.max_len,
              "evidence_in_prompt": not args.no_evidence, "databases": len(dbs), "val_databases": sorted(val_dbs),
              "train_examples": len(train), "val_examples": len(val), **dict(sorted(stats.items()))}
    if tok is not None and records:
        report["mean_tokens"] = round(stats["tokens"] / len(records), 1)
    (PROCESSED / "sft_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "val_databases"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
