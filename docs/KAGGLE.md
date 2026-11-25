# Running the GPU steps on Kaggle (no Hugging Face account needed)

| File | Path |
|---|---|
| Package (upload as a dataset) | `D:\TuringSQL\dist\turingsql_kaggle.zip` |
| Notebook 1: BIRD prep (CPU) | `D:\TuringSQL\notebooks\kaggle_1_prepare_bird.ipynb` |
| Notebook 2: train, merge, evaluate (T4) | `D:\TuringSQL\notebooks\kaggle_2_train_E3.ipynb` |

The public Qwen base model and tokenizer are downloaded anonymously. Nothing is uploaded anywhere: every output
stays in the notebook's **Output** tab.

## 1. Dataset
1. Go to kaggle.com/datasets and click **+ New Dataset**.
2. Upload the zip. Name it `turingsql-kaggle` and make it **Private**.
3. After any code change, open the dataset page and use **New Version** to upload the rebuilt zip
   (`python scripts/package_for_kaggle.py`).

## 2. Notebook 1 (CPU, about 1 hour)
1. Import the notebook (*File → Import Notebook*).
2. Settings: Accelerator **None**, Internet **On**. Input: `turingsql-kaggle`.
3. Click **Save Version → Save** (Save & Run All is the default). The run continues in the background.
4. When it finishes, the end of the log shows `train_examples` and `val_examples`.

## 3. Notebook 2 (T4)
1. Import the notebook.
2. Settings: Accelerator **GPU T4** (x2 is fine; one GPU is used), Internet **On**, Persistence **Files only**.
3. Inputs: `turingsql-kaggle` (its latest version) **and** notebook 1 (*Add Input → Your Work*).
4. Optional: add a `WANDB_API_KEY` secret for live charts. Nothing else is needed.
5. Run the cells one by one, up to and including the **smoke run**. Check two things:
   - the "using SFT files from ..." line;
   - the `s/step` value. Estimate the full run as
     `hours ≈ (2 × train_examples ÷ 16) × s/step ÷ 3600`.
6. Click **Save Version → Save**. This is the full background run.
   - Training stops cleanly after 10 hours (`MAX_HOURS`) at the latest, so the checkpoint is saved.
   - If the run prints `NOT FINISHED`:
     1. Open notebook 2 and choose *Add Input → Your Work → notebook 2* to attach its last version.
     2. Click **Save Version** again. Training continues from the checkpoint; the smoke run is skipped.
7. When it finishes, the **Output** tab holds:
   - `results.zip`: every evaluation run and the training log;
   - `turingsql-1.5b/`: the merged fine-tuned model, about 3 GB.

## 3b. Notebook 3 (T4, about 2 to 3 hours): merge and evaluate
Use this when notebook 2 trained but did not merge or evaluate, or to redo the evaluations without retraining.
1. Rebuild and re-upload the package if the code changed (`python scripts/package_for_kaggle.py`, then **New Version**
   on the dataset). Notebook 2 and 3 must see the version with the fixes below.
2. Import `kaggle_3_merge_eval.ipynb`. Settings: Accelerator **GPU T4**, Internet **On**.
3. Inputs: the dataset's **latest** version and notebook 2 (*Add Input → Your Work*). It finds the adapter at
   `ckpt/E3_seed0/final_adapter` in notebook 2's output.
4. **Save Version → Save & Run All**. It merges, checks 50 prompts give identical output, then runs E3, E0, E1 and
   E5. A failed evaluation is listed at the end instead of stopping the run, and re-running skips finished runs.

Two failures hit the first run of notebook 2 (both fixed in the code, so keep the package up to date):
- **Merge: `ImportError: Found an incompatible version of torchao`.** Kaggle ships `torchao` 0.10, but the newest
  `peft` needs 0.16 or later. The notebooks run `pip uninstall -y torchao` after installing the libraries.
- **BIRD evaluation: `sqlglot.errors.TokenError`.** Model-written SQL with a backslash-escaped quote crashed the
  scorer and lost the run. `check_sql` now treats any unparseable SQL as a wrong answer (regression test in
  `tests/test_safety.py`).

The "Could not find cuda drivers" line in these logs comes from an unrelated library; the GPU is used.

## 4. Back on the laptop
1. Unzip `results.zip` into `D:\TuringSQL\results\`, then run:
   ```powershell
   .\.venv\Scripts\python.exe scripts\results_table.py
   ```
2. Download the `turingsql-1.5b` folder from the Output tab into `D:\TuringSQL\models\turingsql-1.5b\`. This is
   done in the browser; no token is needed. Chrome may open JSON files in a tab or save them as `.txt`: rename them
   back (`config.json`, `tokenizer.json`, `vocab.json`, `tokenizer_config.json`, `special_tokens_map.json`,
   `added_tokens.json`, `generation_config.json`, `merge_report.json`; `chat_template.jinja`). Eleven files in
   total, `model.safetensors` is 3.09 GB. The Kaggle command line (`kaggle kernels output <user>/<notebook>`) is the
   alternative and needs an API token.
3. The final Olist test evaluation (`--final`) is run **once**, at the very end.
