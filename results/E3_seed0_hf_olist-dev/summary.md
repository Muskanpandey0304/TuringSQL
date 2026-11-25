# Run `E3_seed0_hf_olist-dev`

- Model: `/kaggle/working/turingsql-1.5b` via `hf`; dataset `olist-dev`; prompt v1
- Definitions in prompt: False; few-shot examples: 0; sample rows: 2; concurrency 1; max_tokens 256
- Created 2026-11-25T06:44:15Z on Linux x86_64 py3.13.16; git None

| Metric | Value |
|---|---|
| **Execution accuracy (strict)** | **25.0% [12.5, 40.0] (n=40)** |
| Execution accuracy (relaxed, extra columns allowed) | 25.0% [12.5, 40.0] (n=40) |
| Valid SQL rate | 72.5% |
| Latency mean / p50 / p95 | 2.876 / 2.68 / 6.397 s |
| Wall time | 119.3 s |
| Mean prompt tokens | 1716.6 |
| Share of prompt tokens served from prefix cache | None |

| Level | EX strict, 95% CI |
|---|---|
| definition | 0.0% [0.0, 0.0] (n=5) |
| easy | 80.0% [50.0, 100.0] (n=10) |
| hard | 0.0% [0.0, 0.0] (n=12) |
| medium | 15.4% [0.0, 38.5] (n=13) |

Errors: exec_error 11

95% intervals: percentile bootstrap over questions, 2,000 resamples, seed 0.
