"""Retrieve glossary definitions for each Olist question (experiment E5).

Writes data/olist/definitions_retrieved_<split>.json: {question_id: [definition text, ...]} and a readable
data/olist/definitions_retrieved_<split>.md listing what each question got, with reranker scores.

    python scripts/retrieve_definitions.py --split dev --k 3 --min-prob 0.05
    python scripts/retrieve_definitions.py --split test --final ...   # only with the frozen settings

--sweep prints, for the dev split, how many definitions each threshold would let through, to choose
--min-prob on dev only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turingsql.benchmark import OLIST_DIR, load_questions  # noqa: E402
from turingsql.retrieval.glossary import GlossaryRetriever  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min-prob", type=float, default=0.05)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        sys.exit("the test split is only used with --final and settings frozen on dev")

    questions = load_questions(args.split)
    r = GlossaryRetriever.from_jsonl(OLIST_DIR / "glossary.jsonl")
    hits = {q["id"]: r.search(q["question"], k=args.k, min_prob=0.0) for q in questions}

    if args.sweep:
        print("min_prob  definition-qs-with-any  other-qs-with-any  mean-defs-per-q")
        defn = [q["id"] for q in questions if q["level"] == "definition"]
        other = [q["id"] for q in questions if q["level"] != "definition"]
        for t in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
            kept = {i: [h for h in hs if h.score >= t or h.term_match] for i, hs in hits.items()}
            print(f"{t:8.2f}  {sum(bool(kept[i]) for i in defn):>5}/{len(defn):<17}"
                  f"{sum(bool(kept[i]) for i in other):>5}/{len(other):<13}"
                  f"{sum(map(len, kept.values())) / len(kept):.2f}")

    out = {i: [h.text for h in hs if h.score >= args.min_prob or h.term_match] for i, hs in hits.items()}
    (OLIST_DIR / f"definitions_retrieved_{args.split}.json").write_text(
        json.dumps({"k": args.k, "min_prob": args.min_prob, "definitions": out}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    lines = [f"# Retrieved definitions ({args.split}, k={args.k}, min_prob={args.min_prob})", ""]
    for q in questions:
        lines.append(f"- **{q['id']}** ({q['level']}) {q['question']}")
        for h in hits[q["id"]]:
            mark = " (term match)" if h.term_match else ("" if h.score >= args.min_prob else " ~~(below gate)~~")
            lines.append(f"  - `{h.id}` {h.score:.3f}{mark}")
    (OLIST_DIR / f"definitions_retrieved_{args.split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote definitions for {len(out)} questions; {sum(bool(v) for v in out.values())} got at least one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
