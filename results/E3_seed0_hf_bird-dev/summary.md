# Run `E3_seed0_hf_bird-dev`

- Model: `/kaggle/working/turingsql-1.5b` via `hf`; dataset `bird-dev`; prompt v1
- Definitions in prompt: False; few-shot examples: 0; sample rows: 2; concurrency 1; max_tokens 256
- Created 2026-11-25T06:41:56Z on Linux x86_64 py3.13.16; git None

| Metric | Value |
|---|---|
| **Execution accuracy (strict)** | **24.4% [20.6, 28.0] (n=500)** |
| Execution accuracy (relaxed, extra columns allowed) | 24.8% [20.8, 28.6] (n=500) |
| Valid SQL rate | 66.0% |
| Latency mean / p50 / p95 | 2.272 / 2.189 / 3.883 s |
| Wall time | 1149.2 s |
| Mean prompt tokens | 1299.9 |
| Share of prompt tokens served from prefix cache | None |

| Level | EX strict, 95% CI |
|---|---|
| all | 24.4% [20.6, 28.0] (n=500) |

Errors: exec_error 157, unsafe 12, too_many_rows 1

95% intervals: percentile bootstrap over questions, 2,000 resamples, seed 0.
