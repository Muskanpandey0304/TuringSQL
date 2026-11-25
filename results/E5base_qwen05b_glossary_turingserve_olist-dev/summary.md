# Run `E5base_qwen05b_glossary_turingserve_olist-dev`

- Model: `qwen-coder-0.5b` via `openai`; dataset `olist-dev`; prompt v1
- Definitions in prompt: True; sample rows: 2; concurrency 4; max_tokens 256
- Created 2026-11-24T08:06:59Z on Windows AMD64 py3.10.11; git None
- Notes: Base 0.5B zero-shot + retrieved glossary definitions (k=3, gate 0.1 or term match), TuringServe CPU

| Metric | Value |
|---|---|
| **Execution accuracy (strict)** | **22.5% [10.0, 37.5] (n=40)** |
| Execution accuracy (relaxed, extra columns allowed) | 22.5% [10.0, 37.5] (n=40) |
| Valid SQL rate | 32.5% |
| Latency mean / p50 / p95 | 60.904 / 45.686 / 165.605 s |
| Wall time | 630.3 s |
| Mean prompt tokens | 1752.1 |
| Share of prompt tokens served from prefix cache | 0.9427 |

| Level | EX strict, 95% CI |
|---|---|
| definition | 0.0% [0.0, 0.0] (n=5) |
| easy | 70.0% [40.0, 100.0] (n=10) |
| hard | 0.0% [0.0, 0.0] (n=12) |
| medium | 15.4% [0.0, 38.5] (n=13) |

Errors: exec_error 25, too_many_rows 1, unsafe 1

95% intervals: percentile bootstrap over questions, 2,000 resamples, seed 0.
