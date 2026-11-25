# Decisions

Why each choice was made, logged as the project is built. Newest at the bottom.

## 2026-11-24

1. **Base model: Qwen2.5-Coder-1.5B-Instruct (0.5B for debugging).** Code-tuned, small enough for QLoRA on a free
   T4, and the Qwen2 architecture is the only one TuringServe loads. Both sizes were checked: `Qwen2ForCausalLM`,
   tied embeddings.
2. **Develop on native Windows, not WSL.** The WSL disk failed to attach (access denied). TuringServe's CPU reference
   path runs fine on Windows with Python 3.10 (Day-1 check: 3/3 greedy outputs identical to HF).
3. **Every decoding setting is passed explicitly.** Qwen's `generation_config.json` sets `repetition_penalty: 1.05`,
   which HF applies even with `do_sample=False`. Comparisons must not inherit model defaults.
4. **One prompt module (`turingsql/prompt.py`, PROMPT_VERSION v1).** Training, evaluation and serving import the
   same template, so a model is never evaluated on a different prompt than it was trained on.
5. **Schema text is built from PRAGMA info, not the stored CREATE statements.** The stored SQL varies in formatting
   between databases. A normalised form is consistent across Spider, BIRD and Olist, and a stable prefix helps the
   prefix cache.
6. **Two safety layers for executing model SQL.** A sqlglot check (exactly one read-only query, no write, DDL,
   ATTACH or PRAGMA nodes) and a read-only connection with a timeout and a row cap. Either layer alone would stop a
   write.
7. **Statistics ported from AdaptiveRAG (commit 62fb23a)**, so both projects report CIs and McNemar tests the same way.
8. **Olist benchmark database = 7 frozen mart tables** (`data/olist/olist_marts.db`). Staging tables are hidden. The
   frozen state is a **content hash** of the sorted rows (`marts_manifest.json`), not a file hash, because SQLite file
   bytes can change while the data stays the same. `tests/test_olist_gold.py` checks that it reproduces the Olist
   README numbers.
9. **Gold SQL rules:**
   - Gold SQL never rounds; the grader handles numeric precision.
   - Questions state units ("as a percentage") and what to return.
   - Medium and hard questions say "delivered orders" explicitly (`is_delivered`).
   - Only the 15 `definition` questions rely on business terms (revenue, repeat rate, late rate, AOV, active
     customer), which use the analysis-window flags (`is_revenue_order` / `is_revenue_item`) exactly as Olist's
     `metrics_definitions.md` defines them.
10. **Top-N questions are checked for ties at the cut-off.** A tie would make the gold answer depend on arbitrary row
    order. The validator re-runs the query without `LIMIT` and compares the sort keys at the boundary.
11. **Split: 40 dev / 80 test, stratified by level, fixed seed 20261124.** This is done only after the review, and the
    test file is frozen by its sha256 at the same time.
12. **Grading rules for execution accuracy (`turingsql/eval/execution.py`):**
    - Rows are compared as a multiset unless the question needs an order.
    - Columns may be permuted.
    - **Strict** (the headline score): same number of columns as the gold answer. **Relaxed** (also reported): extra
      columns are allowed.
    - Numbers match within 1e-6 relative (float noise). A predicted float with 1-4 decimals also matches the gold
      value rounded to that many decimals. A whole number never matches a non-whole value.
    - This was checked on the benchmark itself: gold vs gold = 120/120. Each gold answer against the other 119 gives
      zero strict matches and exactly two relaxed matches. Both are true supersets (H02 contains M03, M28 contains
      M05).
13. **Frozen files are hashed by parsed JSON content** (`turingsql/benchmark.py`), not file bytes, so git
    line-ending conversion between Windows and Linux CI can't break the freeze. `load_questions()` refuses to load
    an edited file.
14. **Second review of the questions (2026-11-24) changed 12 questions before the freeze.** The details are in the
    data quality log and lab notes. The main ones:
    - E30 had two valid readings (7,827 vs 6,535).
    - E16 and E25 changed their answer depending on edge cases.
    - H04 had a tie in Sergipe.
    - H20 depended on whether cancelled orders count.
    - Four questions now say their 0/1 flag format, and three exclude the one order with no payment type.
15. **Prompt schema = 2 sample rows, values cut at 24 characters, floats to 6 significant digits** (about 1,660
    tokens for Olist instead of 2,659). Decided before any model was evaluated, so PROMPT_VERSION stays v1.
16. **Training data is pre-rendered.** `build_sft.py` writes chat messages with the schema already inside, using the
    same `prompt.py` as evaluation. Training on Kaggle then needs one small JSONL file, not the databases.
17. **Validation = 5% of training databases, held out whole** (seed 20261124). Validation loss then measures
    unseen schemas, the same situation as Spider/BIRD dev.
18. **Plain transformers `Trainer` + peft, not TRL.** Completion-only masking is done explicitly in
    `tokenize_example` and unit-tested. Qwen's chat template has no `{% generation %}` markers, which TRL's
    assistant-only loss needs. The model is trained on `<|im_end|>`, so it learns to stop, but not on the newline
    after it.
19. **Examples over 2,048 tokens are dropped, never truncated.** A cut schema or a cut SQL teaches wrong things.
20. **BIRD train is processed one database at a time on Kaggle** (32 GB unzipped does not fit on the laptop). BIRD
    dev databases stay local for evaluation.
21. **Glossary gate = reranker probability >= 0.1 OR verbatim term match**, and reranking uses the plain business
    wording (lab notes, 2026-11-24 afternoon). The glossary has one entry per row of the Olist project's
    `metrics_definitions.md`, written from that file and not per question.
22. **The Olist test split is never packaged for Kaggle** (`package_for_kaggle.py` asserts it). Final test numbers
    are produced locally with `--final`.
23. **E1 few-shot = 3 examples from Spider train, chosen by BM25 on the question, at most one per database.** Each
    example shows only the CREATE TABLE lines of the tables its SQL uses, so 3 examples add about 300 tokens, not
    about 1,800. Shots come from training data only, never from the Olist questions.
24. **"Ask Olist" adds LIMIT 1000 when the model's SQL has none, and always shows the SQL.** The answer table comes
    from the same read-only, single-SELECT path as the evaluator. Example questions in the app are new wording,
    not copies of benchmark questions.
25. **TuringServe serving benchmark: no change to the TuringServe repo.** Its existing `rag` workload reads
    `{system, prompt}` JSONL from `$TS_RAG_PROMPTS`; `scripts/export_bench_prompts.py` writes TuringSQL prompts in
    that format: all 40 Olist dev prompts (shared schema, the cache-friendly case) plus 160 Spider dev prompts over
    20 schemas.
26. **E4 (schema pruning) is deprioritised**, based on measurement (lab notes, "Schema pruning"): every BIRD dev
    schema fits in 3,014 tokens, and BM25 table pruning only saves about 12% of the text at 97% recall. The code
    and the recall script stay, for databases larger than these benchmarks.
