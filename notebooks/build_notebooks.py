"""Generate the Kaggle notebooks from plain Python cell lists (keeps them reviewable in diffs).

    python notebooks/build_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def nb(cells: list[tuple[str, str]]) -> dict:
    out = []
    for kind, src in cells:
        lines = src.strip("\n").splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        out.append(cell)
    return {"cells": out, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                      "language": "python"},
                                       "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


SETUP = r'''
# Code + data are uploaded as a private Kaggle dataset built by scripts/package_for_kaggle.py
import glob
import json
import os
import subprocess
import sys
import zipfile
import shutil

WORK = "/kaggle/working/TuringSQL"
if not os.path.isdir(WORK):
    # Kaggle usually unzips uploaded archives; handle both the zip and the unpacked folder.
    zips = glob.glob("/kaggle/input/**/turingsql_kaggle.zip", recursive=True)
    trees = [os.path.dirname(os.path.dirname(p))
             for p in glob.glob("/kaggle/input/**/turingsql/__init__.py", recursive=True)]

    def built(tree):  # newest package wins; copies without a stamp (older code) lose
        f = os.path.join(tree, "PACKAGE_INFO.json")
        return json.load(open(f))["built_utc"] if os.path.isfile(f) else ""

    trees.sort(key=built, reverse=True)
    if zips:
        zipfile.ZipFile(zips[0]).extractall(WORK)
    elif trees:
        print("code from", trees[0], built(trees[0]) or "(no build stamp)")
        shutil.copytree(trees[0], WORK)
    else:
        raise FileNotFoundError("attach the turingsql-kaggle dataset (dist/turingsql_kaggle.zip)")
os.chdir(WORK)
sys.path.insert(0, WORK)
print(os.listdir(WORK))
'''

PREP = nb([
    ("markdown", """
# TuringSQL 1/2: prepare BIRD train (CPU session, no GPU quota)

BIRD's training databases are 32 GB unzipped, too big for the laptop. This notebook downloads the official
`train.zip`, processes the databases **one at a time** (extract, check every gold query, write the schema
text, delete), then builds the fine-tuning files from Spider + BIRD.

Output: `/kaggle/working/TuringSQL/data/processed/` (sft_train.jsonl, sft_val.jsonl and reports). Save the
notebook version, then attach its output to notebook 2.

Settings: Accelerator **None**, Internet **On**.
"""),
    ("code", SETUP),
    ("code", r'''
!pip -q install "sqlglot>=25" "transformers>=4.45,<5"
!df -h /kaggle/working /tmp | cat
'''),
    ("code", r'''
# Official BIRD train set (CC BY-SA 4.0), about 8.9 GB
!mkdir -p /tmp/bird
if not os.path.isfile("/tmp/bird/train.zip"):
    !wget -q --show-progress -O /tmp/bird/train.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip
!ls -la /tmp/bird
'''),
    ("code", r'''
!python data_prep/prepare_public.py bird-train --zip /tmp/bird/train.zip --workdir /tmp/bird/work
'''),
    ("code", r'''
# Tokenizer only (small download), used to drop examples longer than 2,048 tokens
!python data_prep/build_sft.py --sources spider-train bird-train --tokenizer Qwen/Qwen2.5-Coder-1.5B-Instruct
'''),
    ("code", r'''
# Keep only the small processed files as notebook output; free the scratch space
!rm -rf /tmp/bird/work
!ls -la data/processed && du -sh data/processed
'''),
])

TRAIN = nb([
    ("markdown", """
# TuringSQL 2/2: QLoRA fine-tune, merge, evaluate (T4), no Hugging Face account needed

Experiment **E3** (PROJECT_PLAN.md section 6): Qwen2.5-Coder-1.5B-Instruct, 4-bit NF4, LoRA r=16 on all linear
layers, fp16 (T4 has no bf16), loss on the SQL only. The public base model is downloaded anonymously; nothing
is uploaded anywhere. Everything (trained model, checkpoints, results) is saved in this notebook's **Output**.

**Settings:** Accelerator **GPU T4** (x1 or x2; only one GPU is used), Internet **On**.
**Inputs:** the `turingsql-kaggle` dataset **and** the output of notebook 1.

**If a run stops before training finishes**, it saves a checkpoint and says so. To continue:
1. Add **this notebook's own previous version** as an input (*Add Input -> Your Work -> this notebook*).
2. Run *Save & Run All* again. Training continues from the checkpoint.
"""),
    ("code", SETUP),
    ("code", r'''
# Kaggle's T4 option comes as "T4 x2"; the training script is single-GPU (4-bit + DataParallel breaks),
# so every subprocess below sees only the first GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
!nvidia-smi
!pip -q install "transformers>=4.45,<5" "peft>=0.13" "bitsandbytes>=0.44" "accelerate>=1.0" "sqlglot>=25" rank_bm25
!pip -q uninstall -y torchao
import torch; print(torch.__version__, torch.cuda.get_device_name(0))
'''),
    ("code", r'''
# Use notebook 1's output (Spider + BIRD train) instead of the Spider-only files in the package.
# Pick it by its report, not by path: the package also contains a (Spider-only) sft_report.json.
reports = glob.glob("/kaggle/input/**/data/processed/sft_report.json", recursive=True)
with_bird = [r for r in reports if "bird-train" in json.load(open(r))["sources"]
             and os.path.isfile(os.path.join(os.path.dirname(r), "sft_train.jsonl"))]
if with_bird:
    d = os.path.dirname(with_bird[0])
    for f in os.listdir(d):
        if f.startswith(("sft_", "prep_report_bird-train", "schemas_bird-train")):
            shutil.copy(os.path.join(d, f), "data/processed/")
    print("using SFT files from", d)
else:
    print("WARNING: notebook 1 output not attached; training on Spider only")
print(open("data/processed/sft_report.json").read()[:1200])
'''),
    ("code", r'''
SEED = 0
BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUT = f"/kaggle/working/ckpt/E3_seed{SEED}"
MAX_HOURS = 10.0   # stop and save well before Kaggle's 12 h limit (merge + evaluation need about 1.5 h after)
# A previous version of THIS notebook, attached as input, holds checkpoints to continue from
prev = [os.path.dirname(p) for p in glob.glob(f"/kaggle/input/**/ckpt/E3_seed{SEED}/checkpoint-*/trainer_state.json",
                                              recursive=True)]
RESUME = os.path.dirname(prev[0]) if prev else ""
print("resume from:", RESUME or "nothing (fresh start)")
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY"); REPORT = "wandb"
except Exception:
    REPORT = "none"   # optional live charts only; training does not need it
'''),
    ("code", r'''
# Smoke run (skipped when resuming): 30 steps. Read the s/step line and estimate the full run time.
if not RESUME:
    !python scripts/train.py --model {BASE} --train data/processed/sft_train.jsonl --val data/processed/sft_val.jsonl --out /kaggle/working/ckpt/smoke --max-steps 30 --eval-steps 30 --save-steps 30 --val-limit 50
    !rm -rf /kaggle/working/ckpt/smoke
'''),
    ("code", r'''
# Full run: 2 epochs, checkpoint every 250 steps, clean stop after MAX_HOURS
resume_flag = f"--resume-from {RESUME}" if RESUME else ""
!python scripts/train.py --model {BASE} --train data/processed/sft_train.jsonl --val data/processed/sft_val.jsonl --out {OUT} --epochs 2 --seed {SEED} --report-to {REPORT} --max-hours {MAX_HOURS} {resume_flag}
meta = json.load(open(f"{OUT}/train_meta.json"))
FINISHED = meta["finished"]
print(f"finished={FINISHED}  step {meta['global_step']}/{meta['max_steps']}  best eval loss {meta['best_eval_loss']}")
if not FINISHED:
    print("NOT FINISHED: attach this version as an input and run Save & Run All again to continue.")
'''),
    ("code", r'''
# Merge the adapter into fp16 weights (TuringServe needs merged weights) and check outputs are unchanged
if FINISHED:
    !python scripts/merge.py --adapter {OUT}/final_adapter --out /kaggle/working/turingsql-1.5b --check-prompts data/processed/sft_val.jsonl --n-check 50 --dtype float16 --device cuda
'''),
    ("code", r'''
# All runs: same prompt template, greedy, fp16. Spider/BIRD dev use a fixed 500-question sample (seed 0).
FT = "/kaggle/working/turingsql-1.5b"
RUNS = [
    # (run tag, model, extra evaluate.py flags)
    ("E0_qwen15b", BASE, ""),                 # zero-shot base
    ("E1_qwen15b_3shot", BASE, "--shots 3"),  # 3 retrieved Spider-train examples
    ("E3_seed0", FT, ""),                     # fine-tuned
]
def evaluate(tag, model, flags, ds):
    run_id = f"{tag}_hf_{ds}"
    cmd = (f"python scripts/evaluate.py --engine hf --model {model} --device cuda --dtype float16 "
           f"--dataset {ds} {'--limit 500' if ds != 'olist-dev' else ''} {flags} "
           f"--run-id {run_id} --out /kaggle/working/results")
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
    print(open(f"/kaggle/working/results/{run_id}/summary.md").read())

if FINISHED:
    for tag, model, flags in RUNS:
        for ds in ["spider-dev", "bird-dev", "olist-dev"]:
            evaluate(tag, model, flags, ds)
    # E5: fine-tuned + retrieved glossary on Olist; BIRD dev with its evidence (the public version of a glossary)
    evaluate("E5_seed0_glossary", FT, "--definitions", "olist-dev")
    evaluate("E5_seed0_evidence", FT, "--definitions", "bird-dev")
'''),
    ("code", r'''
R = "/kaggle/working/results"
if FINISHED:
    pairs = [(f"{a}_hf_{ds}", f"E3_seed0_hf_{ds}") for ds in ["spider-dev", "bird-dev", "olist-dev"]
             for a in ["E0_qwen15b", "E1_qwen15b_3shot"]]
    pairs += [("E3_seed0_hf_olist-dev", "E5_seed0_glossary_hf_olist-dev"),
              ("E3_seed0_hf_bird-dev", "E5_seed0_evidence_hf_bird-dev")]
    for a, b in pairs:
        subprocess.run(f"python scripts/compare.py {R}/{a} {R}/{b}", shell=True, check=True)
        print()
'''),
    ("code", r'''
# Tidy the Output tab: keep checkpoints until training is done, the merged model and the results.
os.chdir("/kaggle/working")
if FINISHED:
    !zip -qr results.zip results ckpt/E3_seed0/train_meta.json && ls -la results.zip
    !rm -rf ckpt/E3_seed0/checkpoint-*    # the merged model replaces them once training is done
!rm -rf TuringSQL
!du -sh /kaggle/working/* | cat
'''),
])


EVAL = nb([
    ("markdown", """
# TuringSQL 3/3: merge the trained E3 adapter and run all evaluations (T4)

Training (notebook 2) finished but its merge and evaluation steps crashed (a `torchao` version clash in
`peft`, and one malformed model-written SQL string that crashed the scorer). Both are fixed in the code
package, so this notebook only **merges, verifies and evaluates**; no training.

**Settings:** Accelerator **GPU T4** (x1 or x2), Internet **On**.
**Inputs:** the **new version** of the `turingsql-kaggle` dataset (with the fixes) **and** the output of
notebook 2 (*Add Input -> Your Work -> turingsql-2-train-E3*). Everything is saved in this notebook's **Output**.

A failed evaluation no longer stops the notebook: it is listed at the end, and re-running skips finished ones.
"""),
    ("code", SETUP),
    ("code", r'''
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
!nvidia-smi
!pip -q install "transformers>=4.45,<5" "peft>=0.13" "bitsandbytes>=0.44" "accelerate>=1.0" "sqlglot>=25" rank_bm25
!pip -q uninstall -y torchao
import torch; print(torch.__version__, torch.cuda.get_device_name(0))
assert torch.cuda.is_available(), "no GPU: set the Accelerator to GPU T4"
'''),
    ("code", r'''
BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
FT = "/kaggle/working/turingsql-1.5b"
R = "/kaggle/working/results"
os.makedirs(R, exist_ok=True)
adapters = glob.glob("/kaggle/input/**/ckpt/E3_seed0/final_adapter/adapter_model.safetensors", recursive=True)
assert adapters, "attach the output of notebook 2 (turingsql-2-train-E3) as an input"
ADAPTER = os.path.dirname(adapters[0])
print("adapter:", ADAPTER)
# keep the E0 Spider result already produced by notebook 2 (do not redo 12 minutes of work)
for d in glob.glob("/kaggle/input/**/results/E0_qwen15b_hf_spider-dev", recursive=True):
    if os.path.isfile(os.path.join(d, "summary.json")) and not os.path.isdir(f"{R}/{os.path.basename(d)}"):
        shutil.copytree(d, f"{R}/{os.path.basename(d)}"); print("reused", d)
# the BIRD/Spider SFT files are only needed for the 50-prompt merge check
reports = glob.glob("/kaggle/input/**/data/processed/sft_val.jsonl", recursive=True)
VAL = reports[0] if reports else "data/processed/sft_val.jsonl"
print("merge-check prompts:", VAL)
'''),
    ("code", r'''
# Merge the adapter into fp16 weights and check greedy outputs are unchanged. Stops the notebook on failure.
r = subprocess.run(f"python scripts/merge.py --adapter {ADAPTER} --out {FT} --check-prompts {VAL} --n-check 50 "
                   "--dtype float16 --device cuda", shell=True, capture_output=True, text=True)
print(r.stdout[-2000:]); print(r.stderr[-3000:])
assert os.path.isfile(f"{FT}/model.safetensors") or glob.glob(f"{FT}/model*.safetensors"), "merge failed, see above"
print(open(f"{FT}/merge_report.json").read())
'''),
    ("code", r'''
RUNS = [
    # (run tag, model, extra evaluate.py flags)
    ("E3_seed0", FT, ""),                     # the main result first, so it exists even if time runs out
    ("E0_qwen15b", BASE, ""),                 # zero-shot base
    ("E1_qwen15b_3shot", BASE, "--shots 3"),  # 3 retrieved Spider-train examples
]
FAILED = []
def evaluate(tag, model, flags, ds):
    run_id = f"{tag}_hf_{ds}"
    if os.path.isfile(f"{R}/{run_id}/summary.json"):
        print("skip (done):", run_id); return
    cmd = (f"python scripts/evaluate.py --engine hf --model {model} --device cuda --dtype float16 "
           f"--dataset {ds} {'--limit 500' if ds != 'olist-dev' else ''} {flags} --run-id {run_id} --out {R}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        FAILED.append(run_id); print("FAILED:", run_id); print(r.stderr[-2500:])
    else:
        print(open(f"{R}/{run_id}/summary.md").read())

for tag, model, flags in RUNS:
    for ds in ["spider-dev", "bird-dev", "olist-dev"]:
        evaluate(tag, model, flags, ds)
evaluate("E5_seed0_glossary", FT, "--definitions", "olist-dev")
evaluate("E5_seed0_evidence", FT, "--definitions", "bird-dev")
print("FAILED runs:", FAILED or "none")
'''),
    ("code", r'''
pairs = [(f"{a}_hf_{ds}", f"E3_seed0_hf_{ds}") for ds in ["spider-dev", "bird-dev", "olist-dev"]
         for a in ["E0_qwen15b", "E1_qwen15b_3shot"]]
pairs += [("E3_seed0_hf_olist-dev", "E5_seed0_glossary_hf_olist-dev"),
          ("E3_seed0_hf_bird-dev", "E5_seed0_evidence_hf_bird-dev")]
for a, b in pairs:
    if os.path.isdir(f"{R}/{a}") and os.path.isdir(f"{R}/{b}"):
        subprocess.run(f"python scripts/compare.py {R}/{a} {R}/{b}", shell=True)
        print()
'''),
    ("code", r'''
os.chdir("/kaggle/working")
!zip -qr results.zip results && ls -la results.zip
!rm -rf TuringSQL
!du -sh /kaggle/working/* | cat
'''),
])


def main() -> None:
    for name, book in (("kaggle_1_prepare_bird.ipynb", PREP), ("kaggle_2_train_E3.ipynb", TRAIN),
                       ("kaggle_3_merge_eval.ipynb", EVAL)):
        (HERE / name).write_text(json.dumps(book, indent=1) + "\n", encoding="utf-8")
        print("wrote", HERE / name)


if __name__ == "__main__":
    main()
