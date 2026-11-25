# TuringSQL-1.5B (E3)

A text-to-SQL model: Qwen2.5-Coder-1.5B-Instruct fine-tuned with QLoRA on Spider and BIRD training questions, then
merged into fp16 weights. The weights are not in this repository (3 GB). Rebuild them with
`notebooks/kaggle_2_train_E3.ipynb` and `notebooks/kaggle_3_merge_eval.ipynb` ([docs/KAGGLE.md](KAGGLE.md)).

## Training
| | |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Method | QLoRA: 4-bit NF4 with double quantisation, fp16 compute (T4 has no bf16) |
| LoRA | r = 16, alpha = 32, dropout 0.05, all linear layers |
| Optimiser | `paged_adamw_8bit`, lr 2e-4, cosine, 3% warmup |
| Batch | 2 per device x 8 accumulation = 16 |
| Data | Spider train + BIRD train, chat format, loss on the SQL tokens only |
| Run | 2 epochs, 1,816 steps, 8.5 hours on one Kaggle T4, seed 0 (one seed only) |
| Loss | final train loss 0.123, best validation loss 0.339 |
| Merge | LoRA merged into fp16 weights; 50 of 50 greedy outputs identical before and after |

## Prompt format
The schema (tables, columns, 2 sample rows) and the question go in the user turn, using the chat template in the
model folder. Decoding is greedy, at most 256 new tokens. See `turingsql/prompt.py`.

## Results (execution accuracy, greedy)
| Dataset | Base, zero-shot | Base, 3-shot | This model |
|---|---|---|---|
| Spider dev (500) | 64.4% | 63.2% | **73.0%** [69.0, 77.0] |
| BIRD dev (500) | 18.4% | 16.6% | **24.4%** [20.6, 28.0] |
| BIRD dev with evidence in the prompt (500) | | | **34.0%** [30.0, 38.2] |
| Olist dev (40) | 32.5% | 17.5% | 25.0% [12.5, 40.0] |

Spider and BIRD gains over the base model are significant (exact McNemar, p = 0.0001 and 0.0003). All runs:
[results/RESULTS.md](../results/RESULTS.md).

## Limits
- **It did not improve on the Olist business questions** (25.0% vs 32.5% for the base model; 40 questions, so the
  difference is within noise). Spider and BIRD training alone did not transfer to that warehouse.
- 24.4% on BIRD without evidence is low in absolute terms; about one answer in three on BIRD is not valid SQL.
- One training seed, so run-to-run noise is not measured. The 500-question dev samples have intervals of about
  plus or minus 4 points.
- English questions, SQLite only. It writes read-only queries; this repo runs them through a read-only checker.
- Intended for research and learning, not for unreviewed use on production data.

## Licences
Base model and training data keep their own licences (Qwen2.5-Coder, Spider, BIRD). Check them before reuse.
