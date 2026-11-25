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

## 4. Back on the laptop
1. Unzip `results.zip` into `D:\TuringSQL\results\`, then run:
   ```powershell
   .\.venv\Scripts\python.exe scripts\results_table.py
   ```
2. Download the `turingsql-1.5b` folder from the Output tab into `D:\TuringSQL\models\turingsql-1.5b\`. This is
   done in the browser; no token is needed.
3. The final Olist test evaluation (`--final`) is run **once**, at the very end.
