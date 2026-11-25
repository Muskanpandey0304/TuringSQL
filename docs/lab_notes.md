# Lab notes

## 2026-11-24: Day-1 check (TuringServe serves Qwen2.5-Coder on CPU)

**Setup:** Windows 11 native (WSL disk failed to attach, see below), Python 3.10 venv at `.venv`, torch 2.6.0+cpu,
transformers 4.57.6, TuringServe installed editable from `D:\TuringServe`.

**Models:**
- Qwen2.5-Coder-0.5B-Instruct: `Qwen2ForCausalLM`, tied embeddings, 24 layers, hidden 896, 2 KV heads.
- Qwen2.5-Coder-1.5B-Instruct: `Qwen2ForCausalLM`, tied embeddings, 28 layers, hidden 1536, 2 KV heads
  (config checked only).

Both use the Qwen2 architecture, so TuringServe's loader supports both.

**Command:**
`turingserve serve --model models/Qwen2.5-Coder-0.5B-Instruct --device cpu --kernels reference --dtype float32 --autopilot off`

**Result:** the server starts, and the OpenAI-compatible chat endpoint works. `scripts/day1_check.py` compares
TuringServe with HF `generate` on 3 Olist-style questions: **3/3 greedy outputs identical**, with prefix caching on
and with it off. Latency is about 3-7 s per query on CPU, similar to HF.

**Bug found in my own check, not in TuringServe:**
- The first comparison showed 1/3 identical.
- The cause is Qwen's `generation_config.json`, which sets `repetition_penalty: 1.05`. HF `generate` applies this
  even with `do_sample=False`, while TuringServe ran plain greedy.
- After passing `repetition_penalty=1.0` to HF, the outputs were 3/3 identical.

**Lesson for the eval harness:** always pass every generation setting explicitly (greedy, `repetition_penalty=1.0`)
and never rely on a model's `generation_config.json`. Otherwise baselines and engines silently run different
decoding.

**Observation (motivation for fine-tuning):**
- The base 0.5B model's SQL runs but is often wrong.
- "Late vs on-time review score" returns only the late average.
- "States with most delivered orders" ignores `fact_orders` and the delivered filter.

**WSL issue (not fixed):**
- `wsl -d Ubuntu` fails with `Failed to attach disk 'D:\WSL\Ubuntu\ext4.vhdx' ... Access is denied`.
- This is a Windows permission problem on the moved disk image and needs an admin fix.
- It doesn't block the CPU work. For the GPU phase, the plan uses Kaggle T4.

## 2026-11-24 (afternoon): evaluation runner, first baselines, data and training pipeline

### Prompt length
The full Olist schema with 3 sample rows was 2,659 tokens, mostly 32-character hex ids and 15-digit coordinates.
With 2 sample rows, values cut at 24 characters and floats printed with 6 significant digits, it is about
1,660 tokens (1,717 with the chat template), and nothing the model needs is lost.

### E0: base Qwen2.5-Coder-0.5B, Olist dev (40), TuringServe CPU
`results/E0_qwen05b_turingserve_olist-dev/`
- **Accuracy:** EX 22.5% [10.0, 37.5]. By level: easy 70%, medium 15%, hard 0%, definition 0%.
- **Failures:** valid SQL only 47.5% of the time. 19 of 21 failures are invented columns or tables
  (`dim_order_items`, `T2.quantity`, `segment` on the wrong table), a schema-linking problem.
- **Serving:** 95.4% of prompt tokens came from TuringServe's prefix cache, because every question shares the
  schema prefix. Mean latency 31 s at concurrency 4 on CPU.

### E5-base: same model + retrieved glossary definitions
`results/E5base_qwen05b_glossary_turingserve_olist-dev/`
- **Accuracy:** EX 22.5%, the same 9 questions right (McNemar p = 1.0). Valid SQL fell to 32.5%.
- **What changed:** the model does copy the definitions (`is_revenue_item`, `is_revenue_order` appear in its SQL),
  but it still invents aliases and tables, so it fails anyway. The base model's bottleneck is schema linking, not
  business knowledge. Retry the glossary after fine-tuning (E5 proper).
- **Latency doubled** because a CPU training smoke test ran at the same time. Don't compare latency between these
  two runs.

### Glossary retrieval (ported from AdaptiveRAG)
- **The reranker failed on SQL-style text.** First version: rerank the SQL-mapped text. bge-reranker scored
  "Revenue: SUM(fact_order_items.price) WHERE ..." at 0.003 for "How much revenue did each region bring in?" and
  ranked unrelated entries above it, even though BM25 + dense RRF had it first.
- **Fix:** rerank on the plain business wording, copied from the Olist project's `metrics_definitions.md`, and send
  the model the column-mapped text. The ranking is now right, but the probabilities are tiny (0.001-0.1), so a
  probability gate alone fails.
- **Final gate:** reranker probability >= 0.1 **or** the glossary term appears verbatim in the question. Chosen on
  dev only. All 5 dev definition questions get their definition; 18 of 35 other questions get at least one, mostly
  the harmless "Order" / "Customer" entries.

### Spider preparation
`data/processed/prep_report_spider-*.json`
- **Train:** 8,625 of 8,659 kept (25 duplicates, 3 gold-SQL errors, 6 with more than 10,000 rows).
- **Dev:** 1,032 of 1,034 kept (2 with more than 10,000 rows).
- **Safety check:** no Spider gold query was rejected; sqlglot parses all of them, so the checker does not
  produce false positives on real SQL.
- **Empty results:** 1,606 train gold queries (18.6%) return an empty result, because many Spider databases have
  empty tables. They are kept: they are still valid SQL to learn from.

### SFT files (Spider only so far)
- 7,894 train / 489 val (val = 5% of databases, held out whole).
- 242 examples (2.8%) are longer than 2,048 tokens and were dropped (big schemas; schema pruning, E4, could
  recover them).
- Mean 592 tokens, so one epoch is about 5.0M tokens.

### Training pipeline checks (CPU, 0.5B)
- **Loss masking:** `tests/test_train_masking.py` checks that only `SQL<|im_end|>` is labelled.
- **`train.py --cpu-smoke`:** 3 steps on 16 examples. Loss 0.52 -> 0.26, eval loss 0.316. Checkpointing, the
  best-checkpoint logic and the adapter save all work.
- **`merge.py`:** 3/3 greedy outputs identical before and after merging.
- **TuringServe loads the merged model** (saved by transformers 4.57): 3/3 outputs identical to HF.
- The full chain train -> merge -> serve works.

### BIRD
- **Disk:** the train databases are 32 GB unzipped (train.zip is 8.9 GB), and the laptop has 25 GB free.
  `prepare_public.py bird-train` therefore extracts one database at a time, and runs in Kaggle notebook 1 (a CPU
  session with fast network).
- **Dev:** dev.zip (346 MB) is downloaded locally.

### BIRD dev preparation
`data/processed/prep_report_bird-dev.json`
- **Kept:** 1,512 of 1,534. Dropped: 19 with more than 10,000 rows, 2 that timed out after 10 s, 1 duplicate.
- **Schema sizes** with our prompt format run from 271 to 3,014 tokens per database (european_football_2 is the
  largest). Every BIRD dev schema fits the model's context without pruning.
- **Disk:** the dev databases take 660 MB once extracted. The nested zip copy was deleted after extraction; the
  original download is kept.

### Schema pruning (E4): measured before building it into prompts
Table-level BM25 pruning with a one-hop foreign-key closure (`turingsql/pruning.py`,
`scripts/pruning_recall.py`). Recall means every table used by the gold SQL survived.

| Dataset | k=3 | k=4 | k=5 | k=6 |
|---|---|---|---|---|
| Spider dev, recall | 98.7% | 99.9% | 100% | 100% |
| Spider dev, schema text kept | 83% | 89% | 93% | 95% |
| BIRD dev, recall | 85.2% | 90.5% | 95.0% | 97.4% |
| BIRD dev, schema text kept | 66% | 75% | 84% | 88% |

**Conclusion:** with compact schemas (2 sample rows), pruning is a bad trade on these benchmarks. At the only
safe setting (BIRD k=6, 97% recall) it saves about 12% of the text, while 2.6% of questions lose a table they
need and so become impossible to answer. E4 is deprioritised. It would only matter for databases much larger than
any in BIRD dev; BIRD train has some, and those are currently dropped by the 2,048-token limit.

### Ask Olist app
- **Pipeline (`turingsql/ask.py`):** retrieval -> prompt -> model -> safety check -> add LIMIT -> read-only
  execution -> chart choice. Unit-tested with a fake model, including refusing a DELETE and leaving the data
  unchanged.
- **Streamlit page (`app/streamlit_app.py`):** checked headlessly with `streamlit.testing`. It renders, and an
  unreachable model server shows an error instead of crashing.

### Kaggle package
- `dist/turingsql_kaggle.zip`: 431 MB, 229 files, no Olist test split (the script checks this).
- The notebooks handle Kaggle's automatic unzipping. Notebook 2 finds notebook 1's SFT files by reading
  `sft_report.json` (it must list `bird-train`), not by path, so it cannot pick up the Spider-only copies by
  mistake.

### E0: base Qwen2.5-Coder-1.5B, Olist dev (40), TuringServe CPU
`results/E0_qwen15b_turingserve_olist-dev/`
- **Accuracy:** EX 32.5% [17.5, 47.5]. By level: easy 70%, medium 38.5%, hard 8.3%, definition 0%. Valid SQL 60%.
- **vs 0.5B:** +10 points, but only 6 vs 2 questions are right for one model and not the other: McNemar p = 0.29,
  not significant on 40 questions.
- **Failures:** 13 of 16 execution errors are "no such column" (e.g. `AVG(price)` on fact_orders, which has
  `items_price`; `region` taken from rfm_segments). Definition questions guess wrong meanings: for repeat rate it
  returned the share of customers seen in the last 30 days.
- **Serving:** prefix cache share 95.4%. Mean latency 89 s at concurrency 4. This is fp32 on CPU while the
  machine was swapping: 0.7 GB RAM free with the 1.5B server (7.5 GB), BIRD prep and VS Code running. These
  latencies are not a TuringServe performance number; T4 numbers come from Kaggle.

## 2026-11-24 (evening): Kaggle notebook 1 (BIRD train prep): success
- **Run:** CPU session, 2,937 s (49 min). The 8.9 GB train.zip downloaded in about 4 minutes; the databases were
  processed one at a time.
- **SFT files** (`sft_report.json`, from Spider + BIRD train):
  - train 14,516 / val 1,121 examples;
  - 206 databases, 10 of them held out for validation (4 BIRD, 6 Spider);
  - 11.76M tokens, 752 tokens per example on average.
- **Kept:** Spider 8,383, BIRD 7,254.
- **Dropped as longer than 2,048 tokens:** Spider 242 and **BIRD 1,625 (18% of usable BIRD)**. These are big BIRD
  schemas. It is a known limit of v1: the model sees fewer examples of very wide databases. Table pruning (E4) or a
  longer max length could recover them.
- **Expected training:** 2 epochs × 14,516 / 16 ≈ 1,815 optimiser steps. To be checked against the smoke run's
  s/step.
