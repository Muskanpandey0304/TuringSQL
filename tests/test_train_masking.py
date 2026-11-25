"""Loss masking in scripts/train.py: only the SQL answer (plus <|im_end|>) is trained on."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "Qwen2.5-Coder-0.5B-Instruct"
sys.path.insert(0, str(ROOT / "scripts"))

pytestmark = pytest.mark.skipif(not MODEL.is_dir(), reason="needs the downloaded Qwen tokenizer")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL)


def test_only_answer_is_labelled(tok):
    from train import IGNORE, tokenize_example

    from turingsql.prompt import build_messages

    msgs = build_messages("CREATE TABLE t (a INT);", "How many rows?", sql="SELECT COUNT(*) FROM t")
    ex = tokenize_example(tok, msgs, max_len=2048)
    labelled = [t for t, lab in zip(ex["input_ids"], ex["labels"], strict=True) if lab != IGNORE]
    assert tok.decode(labelled) == "SELECT COUNT(*) FROM t<|im_end|>"
    assert ex["labels"][: len(ex["labels"]) - len(labelled)] == [IGNORE] * (len(ex["labels"]) - len(labelled))


def test_too_long_is_dropped_not_truncated(tok):
    from train import tokenize_example

    from turingsql.prompt import build_messages

    msgs = build_messages("CREATE TABLE t (a INT);" * 200, "q", sql="SELECT 1")
    assert tokenize_example(tok, msgs, max_len=256) is None
