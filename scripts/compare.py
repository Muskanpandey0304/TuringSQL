"""Compare two evaluation runs on the questions they share (exact McNemar test).

    python scripts/compare.py results/E0_a results/E3_b
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turingsql.eval.stats import paired_comparison  # noqa: E402


def load(run_dir: Path) -> dict[str, bool]:
    lines = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    return {str(r["id"]): bool(r["correct"]) for r in map(json.loads, lines)}


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
    a, b = load(a_dir), load(b_dir)
    r = paired_comparison(a, b)
    shared = [k for k in a if k in b]
    acc_a = sum(a[k] for k in shared) / max(1, len(shared))
    acc_b = sum(b[k] for k in shared) / max(1, len(shared))
    print(f"shared questions: {r['n']}")
    print(f"{a_dir.name}: {100 * acc_a:.1f}%   {b_dir.name}: {100 * acc_b:.1f}%")
    print(f"only {a_dir.name} right: {r['only_a']}   only {b_dir.name} right: {r['only_b']}")
    print(f"exact McNemar p = {r['p_value']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
