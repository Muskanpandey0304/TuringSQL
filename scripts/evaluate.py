"""Evaluate a model on a benchmark split.

Examples:
    # TuringServe (or any OpenAI-compatible server) on the Olist dev set
    python scripts/evaluate.py --engine openai --base-url http://127.0.0.1:8011 --model Qwen2.5-Coder-0.5B-Instruct \
        --dataset olist-dev --run-id E0_qwen05b_olist-dev --concurrency 4

    # Hugging Face transformers in-process
    python scripts/evaluate.py --engine hf --model models/Qwen2.5-Coder-0.5B-Instruct --dataset olist-dev \
        --run-id E0_hf_qwen05b_olist-dev

    # Spider / BIRD (after data_prep): --dataset spider-dev | bird-dev   (--limit N for a quick slice)

The Olist test split is only read with --final, so it cannot be used by accident while tuning.
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
from turingsql.eval.runner import RunConfig, hf_generator, openai_generator, run  # noqa: E402

PROCESSED = ROOT / "data" / "processed"


def load_dataset(name: str, final: bool, with_definitions: bool) -> list[dict]:
    if name in ("olist-dev", "olist-test"):
        split = name.split("-")[1]
        if split == "test" and not final:
            sys.exit("olist-test is the frozen final test set; pass --final only for final reported numbers")
        qs = load_questions(split)
        defs: dict[str, list[str]] = {}
        if with_definitions:
            defs_path = OLIST_DIR / f"definitions_retrieved_{split}.json"
            if not defs_path.is_file():
                sys.exit(f"{defs_path} missing: run scripts/retrieve_definitions.py --split {split} first")
            defs = json.loads(defs_path.read_text(encoding="utf-8"))["definitions"]
        db = OLIST_DIR / "olist_marts.db"
        return [{**q, "db_path": db, "definitions": defs.get(q["id"])} for q in qs]
    path = PROCESSED / f"{name}.jsonl"
    if not path.is_file():
        sys.exit(f"{path} missing: run data_prep/prepare_public.py first")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        ex = json.loads(line)
        ex["db_path"] = ROOT / ex["db_path"]
        if with_definitions and ex.get("evidence"):
            ex["definitions"] = [ex["evidence"]]
        out.append(ex)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["openai", "hf"], required=True)
    ap.add_argument("--model", required=True, help="served model name (openai) or local path (hf)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8011")
    ap.add_argument("--api-key", default="none")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--final", action="store_true", help="allow the frozen olist-test split")
    ap.add_argument("--definitions", action="store_true", help="put definitions / BIRD evidence in the prompt")
    ap.add_argument("--shots", type=int, default=0, help="few-shot examples retrieved from Spider train (E1)")
    ap.add_argument("--limit", type=int, help="evaluate a fixed random subset of N examples (seed 0)")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--n-sample-rows", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    examples = load_dataset(args.dataset, args.final, args.definitions)
    if args.limit and args.limit < len(examples):
        examples = sorted(random.Random(0).sample(examples, args.limit), key=lambda e: str(e["id"]))

    if args.shots:
        from turingsql.fewshot import FewShotSelector

        selector = FewShotSelector.from_processed(PROCESSED, "spider-train")
        for ex in examples:
            exclude = ex.get("db_id") if str(ex.get("source", "")) == "spider" else None
            ex["shots"] = selector.select(ex["question"], k=args.shots, exclude_db=exclude)

    if args.engine == "openai":
        gen = openai_generator(args.base_url, args.model, args.max_tokens, api_key=args.api_key)
    else:
        gen = hf_generator(args.model, args.max_tokens, device=args.device, dtype=args.dtype)

    cfg = RunConfig(run_id=args.run_id, model=args.model, engine=args.engine, dataset=args.dataset,
                    max_tokens=args.max_tokens, concurrency=args.concurrency,
                    n_sample_rows=args.n_sample_rows, use_definitions=args.definitions, n_shots=args.shots,
                    notes=args.notes,
                    extra={"limit": args.limit, "n_examples": len(examples)})
    summary = run(examples, gen, cfg, args.out)
    print((args.out / args.run_id / "summary.md").read_text(encoding="utf-8"))
    return 0 if summary else 1


if __name__ == "__main__":
    raise SystemExit(main())
