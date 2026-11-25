# TuringSQL

Fine-tune a small open model (Qwen2.5-Coder-1.5B) to turn plain-English business questions into SQL, measure it by
**running** the SQL, serve it on my own inference engine [TuringServe](../TuringServe), and point it at the data
warehouse from my [Olist analysis](../HEHE/olist-revenue-leak-analysis).

**Status: work in progress.** The E3 fine-tune (QLoRA on Spider + BIRD, Kaggle T4) is trained, merged and evaluated.
Serving it on TuringServe and the remaining experiments come next ([docs/KAGGLE.md](docs/KAGGLE.md)). Every number
below comes from a committed `results/<run>/summary.md`.

## Results

### Fine-tuning works on the public benchmarks (E3, 500-question samples of the dev sets, greedy, fp16)

| Dataset | E0 base, zero-shot | E1 base, 3-shot | **E3 fine-tuned** | E3 vs E0 (exact McNemar) |
|---|---|---|---|---|
| Spider dev | 64.4% [60.0, 68.8] | 63.2% [59.0, 67.4] | **73.0% [69.0, 77.0]** | +8.6 points, p = 0.0001 |
| BIRD dev | 18.4% [15.0, 21.8] | 16.6% [13.6, 19.8] | **24.4% [20.6, 28.0]** | +6.0 points, p = 0.0003 |
| Olist dev (40 questions) | 32.5% [17.5, 47.5] | 17.5% [7.5, 30.0] | 25.0% [12.5, 40.0] | -7.5 points, p = 0.45 (not significant) |

- Few-shot prompting does not help the base model; fine-tuning does. The share of SQL that runs rose too (Spider
  82.6% to 89.6%, BIRD 60.6% to 66.0%).
- **BIRD with its evidence hint in the prompt (E5):** 34.0% [30.0, 38.2], +9.6 points over E3 (p < 0.0001).
- **Olist did not improve** (E3 25.0%, E3 + glossary 22.5%). With 40 questions every difference is within noise, so
  this is not evidence of harm, but fine-tuning on Spider + BIRD alone clearly did not transfer to the business
  warehouse. Closing that gap is what E7 (a little domain data) is for.
- Merge check: 50/50 greedy outputs identical before and after merging the adapter.

### Olist dev baselines run through TuringServe (40 questions)

| Run | Model | Prompt | Execution accuracy, 95% CI | Valid SQL |
|---|---|---|---|---|
| E0 | Qwen2.5-Coder-0.5B base | zero-shot | 22.5% [10.0, 37.5] | 47.5% |
| E5-base | Qwen2.5-Coder-0.5B base | + retrieved glossary | 22.5% [10.0, 37.5] | 32.5% |
| E0 | Qwen2.5-Coder-1.5B base (the model to fine-tune) | zero-shot | 32.5% [17.5, 47.5] | 60.0% |

All runs are full tables in [results/RESULTS.md](results/RESULTS.md). The base models get 70% of easy questions and
0% of the 5 definition questions. 13 of the 1.5B model's 16 failures are invented columns, which is what
fine-tuning on Spider + BIRD targets. 1.5B vs 0.5B is not yet a significant difference (McNemar p = 0.29 on 40
questions). When TuringServe served this run,
**95% of prompt tokens came from its prefix cache**, because every question shares the same schema prefix.

## How it fits together

```
Spider + BIRD train ──► QLoRA fine-tune (Kaggle T4) ──► merge ──► TuringServe (paged KV, prefix cache)
                                                                     │  OpenAI API
Olist question ──► glossary retrieval (AdaptiveRAG stack) ──► prompt ─┘
                                                                     ▼
                                     safety check ──► read-only SQLite (Olist marts) ──► answer
```

| Reused from | What |
|---|---|
| Olist project | A tested star schema (7 mart tables), metric definitions, and published numbers that the gold answers must reproduce |
| AdaptiveRAG | BM25 + e5 + RRF + bge-reranker retrieval, the confidence-gate idea, bootstrap CIs and McNemar tests |
| TuringServe | Serving (Qwen2 loader, paged KV cache, prefix cache, continuous batching); its `rag` workload runs TuringSQL prompts unchanged |

## The Olist benchmark
- 120 hand-written questions with gold SQL on the Olist marts: 30 easy, 40 medium, 35 hard and 15 needing a
  business definition.
- Split 40 dev / 80 test, stratified, with the test set frozen by hash (`data/olist/FROZEN.json`).
- Every gold query is validated:
  - safe and non-empty;
  - no ties at a `LIMIT`;
  - the right ordering flag;
  - several reproduce the Olist README's published numbers (`tests/test_olist_gold.py`).
- The grader runs both queries and compares results:
  - rows match as a multiset;
  - columns may be permuted;
  - numbers tolerate rounding.
- The grader was itself tested against the benchmark: gold vs gold gives 120/120, and gold vs other questions
  gives 0 strict false matches.

## Setup (CPU, Windows or Linux)

```bash
py -3.10 -m venv .venv            # Linux/macOS: python3.10 -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e . --no-deps
pip install -e ../TuringServe --no-deps   # the serving engine

ruff check . && mypy && pytest -q -m "not slow and not gpu"
```

## Reproduce

```bash
# Data
python data_prep/build_olist_marts.py                       # frozen copy of the Olist marts + content hash
python data_prep/validate_olist_questions.py                # checks all 120 gold queries
python data_prep/prepare_public.py spider --zip data/raw/spider/spider_data.zip
python data_prep/prepare_public.py bird-dev --zip data/raw/bird/dev.zip
python data_prep/build_sft.py --sources spider-train --tokenizer models/Qwen2.5-Coder-0.5B-Instruct

# Baseline through TuringServe
turingserve serve --model models/Qwen2.5-Coder-0.5B-Instruct --served-model-name qwen-coder-0.5b \
    --device cpu --kernels reference --dtype float32 --port 8011
python scripts/evaluate.py --engine openai --model qwen-coder-0.5b --dataset olist-dev \
    --run-id E0_qwen05b_turingserve_olist-dev --concurrency 4

# Glossary experiment
python scripts/retrieve_definitions.py --split dev --k 3 --min-prob 0.1
python scripts/evaluate.py ... --definitions

# Compare two runs (exact McNemar test)
python scripts/compare.py results/<run_a> results/<run_b>
```

GPU training, merging and evaluation: [docs/KAGGLE.md](docs/KAGGLE.md).

## Documentation
- [PROJECT_PLAN.md](PROJECT_PLAN.md): goals, experiments, timeline.
- [docs/DECISIONS.md](docs/DECISIONS.md): why each choice was made.
- [docs/lab_notes.md](docs/lab_notes.md): what was run and what was learned, including failures.
- [docs/data_quality_log.md](docs/data_quality_log.md): every dropped or changed example.

## License

MIT, see [LICENSE](LICENSE). Spider and BIRD are CC BY-SA 4.0. Qwen2.5-Coder is Apache-2.0. The Olist dataset
belongs to Olist (Kaggle licence).
