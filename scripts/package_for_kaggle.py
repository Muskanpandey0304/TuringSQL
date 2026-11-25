"""Bundle what the Kaggle notebooks need into dist/turingsql_kaggle.zip (upload it as a private Kaggle dataset).

Contents: the turingsql package, scripts, data_prep, requirements, the processed Spider/BIRD-dev files, the
Spider and BIRD-dev databases needed for evaluation, and the Olist benchmark (marts DB, dev questions, FROZEN).

The Olist *test* questions are deliberately left out: final test numbers are produced locally with --final,
so nothing on Kaggle can tune against them.

    python scripts/package_for_kaggle.py [--no-databases]
"""
from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE_DIRS = ["turingsql", "scripts", "data_prep"]
INCLUDE_FILES = ["pyproject.toml", "requirements.txt", "requirements-train.txt", "LICENSE",
                 "data/olist/questions_dev.json", "data/olist/FROZEN.json", "data/olist/marts_manifest.json",
                 "data/olist/olist_marts.db", "data/olist/glossary.jsonl",
                 "data/olist/definitions_retrieved_dev.json"]
PROCESSED_GLOBS = ["data/processed/*.jsonl", "data/processed/*.json"]
DB_DIRS = ["data/spider/database", "data/bird/dev_databases"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-databases", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "turingsql_kaggle.zip")
    args = ap.parse_args()

    files: list[Path] = []
    for d in INCLUDE_DIRS:
        files += [p for p in (ROOT / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files += [ROOT / f for f in INCLUDE_FILES if (ROOT / f).is_file()]
    for g in PROCESSED_GLOBS:
        files += list(ROOT.glob(g))
    if not args.no_databases:
        for d in DB_DIRS:
            files += [p for p in (ROOT / d).rglob("*.sqlite") if p.is_file()]
    assert not any(p.name == "questions_test.json" for p in files), "test split must never be packaged"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(set(files)):
            z.write(p, p.relative_to(ROOT).as_posix())
        # Build stamp: the notebooks pick the newest code when several copies are attached as inputs
        # (e.g. notebook 1's output also contains a copy of the code it ran with).
        stamp = {"built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        z.writestr("PACKAGE_INFO.json", json.dumps(stamp))
    size = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({len(set(files))} files, {size:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
