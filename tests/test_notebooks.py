"""The generated Kaggle notebooks are up to date and every code cell is valid Python once IPython shell lines
(`!cmd`, with `\\` continuations) are replaced by `pass`."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ["kaggle_1_prepare_bird.ipynb", "kaggle_2_train_E3.ipynb", "kaggle_3_merge_eval.ipynb"]


def _python_only(src: str) -> str:
    out, continued = [], False
    for line in src.split("\n"):
        if continued:
            continued = line.rstrip().endswith("\\")
            continue
        stripped = line.lstrip()
        if stripped.startswith("!"):
            out.append(line[: len(line) - len(stripped)] + "pass")
            continued = line.rstrip().endswith("\\")
        else:
            out.append(line)
    return "\n".join(out)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_cells_compile(name):
    nb = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            compile(_python_only("".join(cell["source"])), f"{name}:cell{i}", "exec")


def test_notebooks_match_builder(tmp_path):
    before = {n: (ROOT / "notebooks" / n).read_text(encoding="utf-8") for n in NOTEBOOKS}
    subprocess.run([sys.executable, str(ROOT / "notebooks" / "build_notebooks.py")], check=True, capture_output=True)
    after = {n: (ROOT / "notebooks" / n).read_text(encoding="utf-8") for n in NOTEBOOKS}
    assert before == after, "notebooks are stale: run python notebooks/build_notebooks.py"
