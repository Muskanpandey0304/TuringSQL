"""Load the Olist benchmark questions and verify that frozen files have not changed.

The hash covers the parsed JSON (keys sorted, compact), not the file bytes, so line-ending changes
made by git on different platforms do not break it, while any change to a question or gold query does.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLIST_DIR = ROOT / "data" / "olist"
FROZEN = OLIST_DIR / "FROZEN.json"


def content_hash(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_questions(split: str) -> list[dict]:
    """Load 'dev' or 'test'. Raises if a frozen file was edited."""
    if split not in ("dev", "test"):
        raise ValueError(f"unknown split {split!r}")
    path = OLIST_DIR / f"questions_{split}.json"
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    expected = frozen["files"][path.name]
    actual = content_hash(path)
    if actual != expected:
        raise RuntimeError(f"{path.name} changed after it was frozen ({actual} != {expected})")
    questions: list[dict] = json.loads(path.read_text(encoding="utf-8"))
    return questions
