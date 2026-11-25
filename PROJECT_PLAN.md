# TuringSQL: Project Plan

**One line:** fine-tune a small open model to turn plain-English business questions into SQL, measure it honestly,
serve it on my own inference engine (TuringServe), and use it to answer questions about my own data warehouse (Olist).

**The story for recruiters:** *"I fine-tuned a model, served it on an engine I wrote, and pointed it at a warehouse I
built."* Training, serving and use, connected end to end, instead of three separate demos.

> Working name: **TuringSQL**. Folder: `D:\TuringSQL`. Rename freely before the first public commit.

---

## 0. Goals, non-goals and "done"

### Goals
1. **Fine-tune** a small model (Qwen2.5-Coder-1.5B-Instruct) with QLoRA so it writes correct SQL more often than the
   same model with prompting alone.
2. **Measure by execution accuracy:** run the SQL and compare the returned rows. Do not compare SQL text. Use held-out
   sets, confidence intervals and significance tests (the same standard as AdaptiveRAG).
3. **Build a private Olist benchmark:** about 120 hand-written business questions with gold SQL. Public benchmarks
   (Spider, BIRD) may be in the base model's pretraining data, so this is the honest test.
4. **Serve the merged model on TuringServe** on a Kaggle T4 and benchmark it against Hugging Face `generate`
   (and vLLM if it installs). This also produces TuringServe's first real GPU numbers.
5. **Add retrieval for schemas and business definitions**, reusing AdaptiveRAG's hybrid retrieval, and measure whether
   it helps.
6. **Build a small "Ask Olist" demo:** question, then the generated SQL, then a result table and chart. Read-only and safe.

### Non-goals (written down so they don't creep in)
- No SQL dialects other than SQLite in v1.
- No multi-turn conversation in v1.
- No reinforcement learning (GRPO and similar); it goes in "future work" only.
- No 7B+ fine-tuning; the hardware can't do it properly.

### Definition of done (v1.0)
- [ ] Fine-tuned model beats the zero-shot base model on Spider dev, BIRD dev **and** the Olist test set, with 95%
      confidence intervals, and a McNemar test reported for each comparison.
- [ ] Model card and merged weights published on Hugging Face Hub (noting the dataset licences).
- [ ] TuringServe serves the merged model, its GPU test suite passes on a T4, and `benchmarks/REPORT.md` has real numbers.
- [ ] "Ask Olist" demo runs locally and has a public demo (HF Space, or a recorded video if hosting a GPU model costs money).
- [ ] README with a results table, limitations and full reproduction commands.

---

## 1. Hardware and accounts

| Resource | Used for | Notes |
|---|---|---|
| Laptop: GTX 1650 4 GB, i7-9750H | Writing code, data prep, CPU tests, 0.5B debug runs, Ollama baselines | No tensor cores, no bf16. Do not train the 1.5B model here. |
| **Kaggle T4** (free, about 30 GPU h/week, 12 h per session) | All training runs, GPU evaluation, TuringServe benchmarks | T4 = sm_75, which is exactly TuringServe's target. fp16 only (no bf16). |
| Colab T4 (free) | Backup if Kaggle quota runs out | Sessions can drop. Always checkpoint. |
| Hugging Face account | Datasets, model hosting, Space for the demo | Write token stored as a Kaggle secret, never committed |
| MLflow (local file store, as in GuardML) or Weights & Biases (free) | Experiment tracking | W&B is easier across Kaggle sessions; MLflow keeps everything local. **Choose W&B for training runs.** |

**Budget rule:** before every long run, do a 200-step smoke run, read the step time, and estimate the full run time. Record the estimate and the actual time in `docs/lab_notes.md`.

---

## 2. Model choice

| Candidate | Why / why not |
|---|---|
| **Qwen2.5-Coder-1.5B-Instruct (main)** | Code-tuned and Qwen2 architecture, so **TuringServe can load it as is** (its loader supports Qwen2 only, including tied embeddings). QLoRA fits easily on a T4. |
| Qwen2.5-Coder-0.5B-Instruct (debug and draft) | Runs on the laptop, for fast pipeline debugging. Also the **draft model** for TuringServe's speculative decoding. |
| Qwen2.5-Coder-7B-Instruct via Ollama (baseline only) | A strong "just use a bigger model" baseline. Not fine-tuned. |
| Claude API (optional reference row) | Shows the ceiling. Run only on the Olist test set to keep cost small. Clearly labelled as an outside reference. |

**First check (Day 1):** open `config.json` for the 1.5B model and confirm `architectures == ["Qwen2ForCausalLM"]`,
then load it with `turingserve serve --device cpu --kernels reference --dtype float32` and send one request. If this
fails, find out on day 1, not in week 4.

---

## 3. Data

### 3.1 Public datasets

| Dataset | Split | Use | Size | Licence |
|---|---|---|---|---|
| Spider 1.0 | train | Training | about 7,000 pairs, 140 databases | CC BY-SA 4.0 |
| Spider 1.0 | dev | Evaluation (databases not seen in training) | 1,034 | CC BY-SA 4.0 |
| BIRD | train | Training | about 9,400 pairs, 69 databases | CC BY-SA 4.0 |
| BIRD | dev | Evaluation (databases not seen in training) | 1,534 | CC BY-SA 4.0 |

- Both datasets ship SQLite databases, so execution accuracy needs no extra setup.
- BIRD includes an `evidence` field (hints such as "revenue refers to price"). Train **with** evidence, and evaluate
  both with and without it. The without-evidence condition is the realistic one, and it motivates the glossary
  retrieval in phase 5.
- Licence note for the model card: CC BY-SA training data, so publish the weights under a compatible licence and say so.

### 3.2 Data cleaning (`data_prep/`)
1. Execute every gold SQL query against its database. **Drop pairs whose gold SQL errors or times out (>10 s).**
   Log the counts in `docs/data_quality_log.md`. This is the same habit as in the Olist project.
2. Deduplicate exact (question, db) pairs.
3. Normalise whitespace in the SQL. Keep the original casing and aliases.
4. Token-length check: prompt + SQL must be ≤ 2,048 tokens. For long BIRD schemas, use the schema pruning from
   §4.2. Never cut the schema text mid-table.
5. Hold out **5% of training databases** (whole databases, not rows) as a validation split for early stopping.
   Never use dev or test for this.

### 3.3 Olist benchmark (the private test set): link to the Olist project

This is the most important dataset in the project, because no model has seen it.

**Source database:** `D:\HEHE\olist-revenue-leak-analysis\data\olist.db`, the star schema already built and tested
by the Olist pipeline:

| Table | Rows | Role |
|---|---|---|
| `fact_orders` | 99,441 | One row per order: dates, `is_delivered`, `is_late`, `delivery_days`, `review_score`, payment |
| `fact_order_items` | 112,650 | One row per item: `price`, `freight_value`, `distance_km` |
| `dim_customer` | 96,096 | `customer_unique_id`, state, region, IBGE population and income |
| `dim_seller` | 3,095 | Seller state, location, region |
| `dim_product` | 32,951 | Category, size and weight |
| `dim_date` | 852 | Calendar, holidays |
| `rfm_segments` | 93,096 | Customer segments |

**Expose to the model:** only the 7 mart tables above. Hide `stg_*` tables; a real analyst would not query staging.
Copy the mart tables into `data/olist/olist_marts.db` with a script, so the benchmark doesn't change if the Olist
pipeline is rebuilt. Record the file's sha256.

**Writing the questions (≈120):**

| Level | Count | Example |
|---|---|---|
| Easy: one table, one filter or aggregate | 30 | "How many orders were delivered late?" |
| Medium: join, group by, order by | 40 | "Average review score by customer region, highest first" |
| Hard: CTE, window function, date logic, ratios | 35 | "Month-over-month revenue growth in 2018" |
| Business definition required | 15 | "What is our repeat rate?" (needs the definition in `metrics_definitions.md`) |

Rules:
- **Write the test questions first and freeze them** (commit, then record the file's hash in the README) before any
  training or prompt tuning touches Olist.
- Gold SQL comes from the existing `sql/` files where possible (`30_kpi_overview.sql`, `31_monthly_trend.sql`,
  `40_cohort_retention.sql`, …). They were already checked against raw CSVs by the Olist data tests.
- **Gold-answer sanity tests:** several gold answers must reproduce numbers already published in the Olist README:
  revenue **R$13.18M**, **96,203** delivered orders, repeat rate **3.0%**, late rate **8.1%**, late vs on-time review
  **2.57 vs 4.29**. If a gold query doesn't match, the gold query is wrong. Put this in `tests/test_olist_gold.py`.
- Split: **40 dev** (allowed for prompt and threshold choices) / **80 test** (touched only for final numbers).
- Paraphrase check: about 20% of questions use words that are not column names ("money made" rather than "revenue",
  "shoppers" rather than "customers").

**Optional domain-adaptation set (experiment E7):** about 300 additional Olist training pairs, generated by a larger
model and **kept only if the SQL executes and a person spot-checks 1 in 10**. Generated from *different* templates
than the test set. It answers one question: does a small amount of in-domain data help beyond Spider + BIRD?

---

## 4. Prompt format and schema handling

### 4.1 Prompt template (identical for training, evaluation and serving)

```
<|im_start|>system
You write one SQLite query that answers the question. Output only SQL.<|im_end|>
<|im_start|>user
### Schema
CREATE TABLE fact_orders (order_id TEXT, customer_unique_id TEXT, ..., is_late INTEGER, review_score INTEGER);
-- sample rows: ...  (3 rows per table, long values cut to 40 chars)
...
### Definitions            (only when the glossary is on; see phase 5)
Revenue: sum of item price for delivered orders, freight excluded.
### Question
What was total revenue in 2018?<|im_end|>
<|im_start|>assistant
SELECT ...<|im_end|>
```

- Use the model's own chat template through `tokenizer.apply_chat_template`. Never hand-write special tokens in two
  places. Store the template once in `turingsql/prompt.py` and import it everywhere.
- **Loss on the assistant tokens only** (completion-only masking).
- Order the schema deterministically (tables alphabetically, columns in database order). A stable prefix helps
  TuringServe's prefix cache (§7.3).

### 4.2 Schema pruning: link to AdaptiveRAG
Large BIRD databases don't fit well in 2,048 tokens, and long irrelevant schemas hurt small models.
- Build one "document" per **column**: `table.column`, type, sample values and the column description where BIRD
  provides one.
- Rank columns for the question with AdaptiveRAG's retrieval: **BM25 + e5-base-v2 fused with RRF, then the
  bge-reranker**. Port `src/retrievers.py` (or import it as a small package). Don't rewrite it.
- Keep the top-k columns, then add back every primary/foreign key needed to join the chosen tables.
- **Measure schema-linking recall:** the share of tables and columns used by the gold SQL that survive pruning. Target
  ≥ 0.95. If recall drops, the SQL can't be right, so report this number next to EX.

---

## 5. Evaluation harness (build this **before** any training)

`turingsql/eval/`:

| Metric | Definition |
|---|---|
| **EX: execution accuracy (headline)** | The predicted and gold SQL return the same result, compared as multisets of rows. Order matters only if the gold SQL has `ORDER BY`. Floats rounded to 4 decimals. |
| Valid-SQL rate | Share of predictions that parse (sqlglot) and run without error |
| Exact-set match (Spider only) | Reported for comparison with papers only; not the headline |
| Schema-linking recall | See §4.2 |
| Latency | Time to first token and total time per query (from the serving phase) |

Safety in the evaluator (it executes model-written SQL):
- Open SQLite **read-only** (`file:...?mode=ro`, `uri=True`), with a 10 s timeout through `progress_handler` and
  max 10,000 rows fetched.
- Reject anything that sqlglot does not parse as a single `SELECT`/`WITH` statement.

Statistics, reused from AdaptiveRAG (`D:\Rag\src\eval.py`):
- `bootstrap_ci(...)` for 95% CIs on every EX number, with fixed seed 0.
- `mcnemar_exact(only_a, only_b)` for every A-vs-B comparison on the same questions.
- Copy them into `turingsql/eval/stats.py` with a comment pointing to the source, plus unit tests.

Outputs: `results/<run_id>/predictions.jsonl` (question, prediction, gold, correct, error, latency) and a generated
`results/<run_id>/summary.md`. **Every number in the README comes from a committed summary file.** This is the same
rule as TuringServe's `benchmarks/REPORT.md`.

Test the harness itself:
- gold vs gold = 100%;
- a deliberately broken query = 0%;
- row-order and float edge cases;
- a timeout query is stopped.

---

## 6. Experiments

All runs are logged to W&B or MLflow with the git commit, config and dataset hashes.

| ID | Experiment | Question it answers |
|---|---|---|
| E0 | Base 1.5B, zero-shot | Starting point |
| E1 | Base 1.5B, 3-shot (examples chosen by retrieval from the train set) | How far does prompting alone go? |
| E2 | Qwen2.5-Coder-7B via Ollama, 3-shot | Is a 5× bigger model better than fine-tuning a small one? |
| **E3** | **QLoRA fine-tune, Spider + BIRD** | **The main result** |
| E4 | E3 + schema pruning (§4.2) | Does retrieval help on big schemas (BIRD)? |
| E5 | E3 + business glossary retrieval (phase 5) | Does it fix the "business definition" questions on Olist? |
| E6 | E3 + one self-correction retry (run the SQL; if it errors, give the error back once) | Cheap gain at inference time? |
| E7 | E3 + 300 Olist synthetic pairs | Does a little domain data help? (Olist test only) |
| E8 | E3 served in fp16 vs INT8 weights (TuringServe `--quantization int8`) | Does INT8 cost accuracy? |
| (ref) | Claude, zero-shot, Olist test only | Ceiling reference, clearly labelled |

### 6.1 Training settings for E3 (starting point; change only with a recorded reason)

| Setting | Value | Why |
|---|---|---|
| Quantisation | 4-bit NF4, double quant, compute dtype **fp16** | T4 has no bf16 |
| LoRA | r = 16, alpha = 32, dropout = 0.05, on all linear layers (q, k, v, o, gate, up, down) | Standard strong default |
| Optimiser | `paged_adamw_8bit`, lr 2e-4, cosine schedule, 3% warmup, weight decay 0 | |
| Batch | per-device 2 × grad accumulation 8 = 16 effective | Fits T4 with gradient checkpointing |
| Max length | 2,048 tokens | |
| Epochs | 2, keeping the best checkpoint by validation loss **and** validation EX (on 200 held-out-db questions) | Loss alone can mislead for SQL |
| Seeds | 2 seeds for the final config | To see run-to-run noise |
| Libraries | `transformers`, `peft`, `trl` (`SFTTrainer`), `bitsandbytes`, `accelerate`; versions pinned in `requirements-train.txt` | |

Kaggle reality:
- Save a checkpoint every 250 steps to `/kaggle/working`, and push adapters to a private HF repo, so a dropped
  session costs minutes, not hours.
- Keep one notebook per run, generated from `scripts/train.py` and committed under `notebooks/`.

### 6.2 After training
1. Merge the LoRA adapter into the fp16 base: `merge_and_unload()`, then `save_pretrained(safe_serialization=True)`.
   **TuringServe has no LoRA support (out of scope in its README), so merging is required.**
2. Verify the merge: the merged model and base+adapter must produce identical greedy outputs on 50 prompts.
3. Push the merged model and the model card to the Hub.

---

## 7. Serving on TuringServe: link to TuringServe

This phase finishes TuringServe's open item ("no T4 numbers exist yet") with a real workload.

### 7.1 Bring-up on Kaggle T4
1. Install TuringServe's pinned stack (torch 2.6.0 + cu124, Triton 3.2.0). Run `scripts/check_versions.py` and
   `scripts/probe_triton_mma.py`.
2. Run `pytest tests/kernels -m gpu -q`. **Fix the kernel failures first.** Record everything in TuringServe's
   `docs/lab_notes.md`.
3. Serve the merged TuringSQL model:
   `turingserve serve --model <merged_dir>`, first with `--kernels reference`, then with Triton kernels and CUDA graphs.
4. Correctness gate: TuringServe's greedy output must equal HF `generate` on the 80 Olist test prompts. **EX must be
   identical.** Any difference is a TuringServe bug to fix before benchmarking.

### 7.2 Benchmark (results go in **both** repos)

| Engine | Setup |
|---|---|
| HF `transformers` `generate` (batch 1 and padded batches) | Baseline everyone understands |
| **TuringServe** | Defaults, then each feature on and off |
| vLLM (if it installs on T4) | The comparison the TuringServe README is built around |

Workload: real TuringSQL prompts (long schema prefix, short SQL output), at concurrency 1, 4 and 16. Measure TTFT,
inter-token latency, throughput (tokens/s and queries/s) and peak memory. Use TuringServe's
`benchmarks/serve_bench.py` with a new `--dataset turingsql` loader, and keep its "only committed JSON becomes a
claim" rule.

### 7.3 Feature studies built for this workload

| Feature | Why it fits text-to-SQL | What to measure |
|---|---|---|
| **Prefix caching** | Every Olist question shares the same long schema prefix, so the KV cache for it is reused | TTFT with caching on and off; cache hit rate from `/metrics` |
| **Speculative decoding, prompt lookup (n-gram)** | SQL copies table and column names from the schema in the prompt, so n-gram drafts should be accepted often | Acceptance rate and speed-up; EX must be unchanged |
| Speculative decoding, 0.5B draft model | Same model family | Same as above |
| INT8 weights | Memory and speed on a T4 | Speed and **EX difference** (experiment E8) |
| Autopilot | Self-tuning under mixed load | Throughput against hand-picked settings |

**Expected headline (to be confirmed, not claimed in advance):** prefix caching cuts TTFT a lot for repeated-schema
queries, and prompt-lookup decoding helps more on SQL than on chat. Whatever the numbers are, report them.

---

## 8. Retrieval for schemas and business definitions: link to AdaptiveRAG

Two retrieval jobs, both built on AdaptiveRAG's code (`retrievers.py`, the RRF fusion and the reranker):

1. **Schema pruning** (§4.2): column-level retrieval for big databases.
2. **Business glossary:**
   - Index Olist's `reports/metrics_definitions.md` (one definition per row: revenue, repeat rate, late rate,
     delivered order, active customer, …).
   - Index the column comments as well.
   - For each question, retrieve the top 3 definitions and put them in the `### Definitions` block.
   - Use the **confidence gate idea from AdaptiveRAG**: if the reranker score is below a threshold, add no
     definitions rather than misleading ones. Set the threshold on Olist dev only.
   - Measure: EX on the 15 "business definition" questions with and without the glossary (E5).
   - BIRD's `evidence` field is the public version of the same idea.
3. **Few-shot example retrieval** for E1: retrieve the 3 most similar training questions with the same hybrid search.

Credit and reuse: copy the needed modules into `turingsql/retrieval/` with a header comment naming AdaptiveRAG and
the commit hash. Keep the MIT licence notice.

---

## 9. "Ask Olist" app: link to Olist

`app/`, a FastAPI backend plus a Streamlit UI (the same stack as GuardML and AdaptiveRAG).

Flow:
```
question -> glossary + schema retrieval -> prompt -> TuringServe (OpenAI API) -> SQL
         -> safety check -> read-only execution -> table + auto chart + "how this was computed"
```

UI shows:
- the generated SQL (editable, with a re-run button);
- the business definitions used;
- the result table and an automatic chart: a line chart for date columns, a bar chart for category plus number;
- the latency, and whether the prefix cache was hit.

Safety, a small version of the GuardML idea:
- read-only connection;
- single `SELECT`/`WITH` only (sqlglot);
- `LIMIT` added if missing;
- 10 s timeout;
- the model's SQL is shown, never executed silently.

Ten example questions are clickable from the Olist README findings: "What share of customers bought twice?",
"Which states have the highest late rate?", "Review score for late vs on-time orders?". Each answer can be checked
against the published Olist numbers.

Deployment:
- **Local:** `docker compose up` runs TuringServe (GPU), the API and the UI.
- **Public:** a Hugging Face Space. A free CPU Space is too slow for a 1.5B model, so use the INT8 or a GGUF build on
  CPU there, or embed a **2-minute demo video** in the README. Say clearly which one the public demo uses.

---

## 10. Repository structure

```
TuringSQL/
  turingsql/
    prompt.py            single source of truth for the prompt template
    schema.py            schema serialisation, sample rows, deterministic order
    retrieval/           ported from AdaptiveRAG: BM25, e5, RRF, reranker, glossary
    eval/                execution accuracy, safety sandbox, stats (bootstrap, McNemar), reports
    safety.py            sqlglot checks, read-only execution
  data_prep/             download, clean and filter Spider/BIRD; build olist_marts.db
  data/olist/            questions_dev.json, questions_test.json (frozen), glossary.jsonl
  scripts/
    train.py             QLoRA SFT (also used by the Kaggle notebooks)
    merge.py             merge the adapter, then verify identical greedy output
    evaluate.py          --model --engine hf|turingserve|ollama|anthropic --set spider|bird|olist
  notebooks/             one Kaggle notebook per training or benchmark run
  app/                   FastAPI + Streamlit "Ask Olist"
  results/               committed summaries; predictions as JSONL
  tests/                 harness, safety, prompt, gold-answer sanity, retrieval
  docs/
    lab_notes.md         every run: config, time, result, what I learned
    DECISIONS.md         why each choice was made (the AdaptiveRAG habit)
    data_quality_log.md  dropped pairs and why
  requirements.txt  requirements-train.txt  pyproject.toml  docker-compose.yml  LICENSE  README.md
```

Engineering standards, the same as the other repos:
- `ruff` and `mypy` clean;
- `pytest` on CPU in GitHub Actions;
- **a regression gate:** CI runs the harness on 30 cached Olist predictions and fails if scoring changes.

---

## 11. Timeline (about 6 weeks, part-time)

| Week | Phase | Deliverables | Done when |
|---|---|---|---|
| 1 | **0. Setup + 1. Data + 5. Harness** | Repo, CI, Day-1 TuringServe load check, Spider/BIRD downloaded and cleaned, eval harness with tests | Harness gives 100% on gold-vs-gold for all three sets |
| 1-2 | **Olist benchmark** | `olist_marts.db`, 120 questions, gold SQL, sanity tests, frozen test hash | Gold answers match the Olist README numbers |
| 2 | **Baselines E0-E2** | First results table with CIs | Numbers committed in `results/` |
| 2-3 | **Fine-tune E3** | Smoke run, full run (2 seeds), merged model, model card | E3 beats E0 on all three sets, or the README explains why not |
| 3-4 | **Retrieval E4, E5, E6** | Schema pruning, glossary, self-correction | Each has a result row and a DECISIONS entry |
| 4-5 | **TuringServe on T4 (§7)** | GPU tests pass, correctness gate, benchmark report, feature studies, E8 | `REPORT.md` in both repos has real numbers |
| 5 | **App (§9)** | Ask Olist, Docker compose, demo video or Space | A stranger can run it from the README |
| 6 | **Write-up** | README, results charts, blog/LinkedIn post, E7 if time | Checklist in §0 is complete |

If behind schedule, cut in this order: E7, vLLM comparison, Autopilot study, public Space (keep the video).

---

## 12. Risks and what to do about them

| Risk | Plan |
|---|---|
| TuringServe GPU kernels fail on T4 | That's real engineering work and good to show. Fall back to `--kernels reference` for the app, and keep debugging in TuringServe's lab notes. The fine-tuning results don't depend on it. |
| Base model has seen Spider/BIRD dev (contamination) | State it openly. The private Olist test set is the honest number, and that's why it exists. |
| Fine-tuning doesn't beat few-shot on Olist | Still a valid result. Report it, check whether E5/E7 close the gap, and explain. |
| Kaggle session timeouts | Checkpoints every 250 steps pushed to HF; resumable training script |
| BIRD schemas too long | Schema pruning (§4.2); log how many examples were truncated |
| Small test sets make differences noisy | CIs + McNemar on every comparison. Call gaps "within noise" when they are. |
| Executing model SQL is unsafe | Read-only, single SELECT, timeout, row cap, all tested |

---

## 13. README outline (write it as results arrive, not at the end)

1. One-paragraph summary and a GIF of Ask Olist.
2. **Results at a glance:** EX table (E0-E8 × Spider/BIRD/Olist) with CIs, plus the serving table (HF vs TuringServe
   vs vLLM).
3. How it works: a diagram linking the three projects.
4. Olist benchmark: how it was built and why it's the honest number.
5. Fine-tuning details and training curves.
6. Serving study: prefix cache, speculative decoding, INT8.
7. Limitations, contamination note, licences.
8. Reproduce everything (exact commands, hardware, versions, seeds).
9. Related projects: links to Olist, AdaptiveRAG and TuringServe.

### Resume bullets (fill in the real numbers)
- Fine-tuned Qwen2.5-Coder-1.5B with QLoRA for text-to-SQL, raising execution accuracy from **X%** to **Y%** on a
  private 80-question business benchmark (95% CI, McNemar p = Z), on free T4 GPUs.
- Served the model on a from-scratch inference engine (paged KV cache, Triton kernels). Prefix caching cut time to
  first token by **N×** and prompt-lookup speculative decoding gave **M×** decode speed-up, with no change in accuracy.
- Added hybrid retrieval (BM25 + dense + reranker) of schema columns and business definitions, improving accuracy on
  definition-dependent questions from **A** to **B**.

---

## 14. How the three projects connect (summary)

| Existing project | What TuringSQL takes from it | What TuringSQL gives back |
|---|---|---|
| **Olist** | Tested star schema (`olist.db`), metric definitions, verified SQL in `sql/`, published numbers for gold-answer checks | A natural-language interface to the warehouse ("Ask Olist") |
| **AdaptiveRAG** | Hybrid retrieval, reranker, confidence gate, `bootstrap_ci`, `mcnemar_exact`, the evaluation discipline | A second, different use of the retrieval stack, with its own measurements |
| **TuringServe** | The serving engine (Qwen2 loader, prefix cache, speculative decoding, INT8, OpenAI API) | A real model and workload, the first T4 GPU numbers, and bugs found under real load |
| (GuardML, optional) | The "safety in code, not in the prompt" idea for executing model SQL | — |

---

## 15. First three tasks (start here)

1. Create the repo skeleton (§10), `pyproject.toml`, CI with ruff, mypy and pytest.
2. Day-1 check: download Qwen2.5-Coder-0.5B-Instruct and 1.5B-Instruct, confirm the Qwen2 architecture, and load the
   0.5B model in TuringServe on CPU with `--kernels reference`.
3. Build `data_prep/build_olist_marts.py`, then start writing the 120 Olist questions. Write the test questions first
   and freeze them before any other work on Olist.

---

## 16. Progress (updated 2026-11-24)

| Phase | State | Evidence |
|---|---|---|
| Day-1 check: TuringServe serves Qwen2.5-Coder | **Done** | lab notes; 3/3 greedy outputs identical to HF |
| Repo skeleton, CI, core modules | **Done** | ruff, mypy, pytest |
| Olist benchmark (120 questions, frozen 40/80 split) | **Done** | `data/olist/FROZEN.json`, `tests/test_olist_gold.py` |
| Evaluation harness (execution accuracy, CIs, McNemar) | **Done** | `turingsql/eval/`, grader checked on the benchmark itself |
| Baselines | **Done.** E0 0.5B and 1.5B on CPU (22.5%, 32.5% Olist dev); E0 and E1 1.5B on Spider, BIRD and Olist on Kaggle | `results/RESULTS.md` |
| Spider prep | **Done** | `data/processed/prep_report_spider-*.json` |
| BIRD prep | dev **done** (1,512 kept); train runs in Kaggle notebook 1 | `docs/KAGGLE.md` |
| Training / merge pipeline | **Built and smoke-tested on CPU** (train -> merge -> TuringServe, outputs identical) | lab notes |
| Fine-tune E3 on T4 | **Done** (1 seed, 1816 steps, 8.5 h). Spider 73.0% vs 64.4%, BIRD 24.4% vs 18.4% (both p < 0.001); Olist dev 25.0% vs 32.5% (n = 40, not significant). Second seed not run. Merged model verified (50/50 identical) | `notebooks/kaggle_2_train_E3.ipynb`, `kaggle_3_merge_eval.ipynb`, `results/RESULTS.md` |
| Glossary retrieval (E5) | Retried after fine-tuning: BIRD with evidence 34.0% (+9.6 points over E3); Olist glossary 22.5% vs 25.0% (no gain, n = 40) | `results/E5_seed0_*` |
| Schema pruning (E4) | Measured, deprioritised (BIRD schemas fit; pruning saves 12% at 97% recall) | lab notes |
| Self-correction (E6), domain data (E7), INT8 (E8) | Not started | |
| TuringServe T4 benchmark | Workload export ready (`scripts/export_bench_prompts.py`) | |
| Ask Olist app | Pipeline and Streamlit UI built and tested headlessly | `turingsql/ask.py`, `app/` |
