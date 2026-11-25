# Run `E5_seed0_glossary_hf_olist-dev`

- Model: `/kaggle/working/turingsql-1.5b` via `hf`; dataset `olist-dev`; prompt v1
- Definitions in prompt: True; few-shot examples: 0; sample rows: 2; concurrency 1; max_tokens 256
- Created 2026-11-25T07:45:48Z on Linux x86_64 py3.13.16; git None

| Metric | Value |
|---|---|
| **Execution accuracy (strict)** | **22.5% [10.0, 37.5] (n=40)** |
| Execution accuracy (relaxed, extra columns allowed) | 22.5% [10.0, 37.5] (n=40) |
| Valid SQL rate | 65.0% |
| Latency mean / p50 / p95 | 3.237 / 2.86 / 8.266 s |
| Wall time | 143.7 s |
| Mean prompt tokens | 1752.1 |
| Share of prompt tokens served from prefix cache | None |

| Level | EX strict, 95% CI |
|---|---|
| definition | 20.0% [0.0, 60.0] (n=5) |
| easy | 60.0% [30.0, 90.0] (n=10) |
| hard | 0.0% [0.0, 0.0] (n=12) |
| medium | 15.4% [0.0, 38.5] (n=13) |

Errors: exec_error 13, timeout 1

95% intervals: percentile bootstrap over questions, 2,000 resamples, seed 0.
