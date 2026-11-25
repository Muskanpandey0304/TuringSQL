# Run `E3_seed0_hf_spider-dev`

- Model: `/kaggle/working/turingsql-1.5b` via `hf`; dataset `spider-dev`; prompt v1
- Definitions in prompt: False; few-shot examples: 0; sample rows: 2; concurrency 1; max_tokens 256
- Created 2026-11-25T06:20:45Z on Linux x86_64 py3.13.16; git None

| Metric | Value |
|---|---|
| **Execution accuracy (strict)** | **73.0% [69.0, 77.0] (n=500)** |
| Execution accuracy (relaxed, extra columns allowed) | 73.2% [69.2, 77.2] (n=500) |
| Valid SQL rate | 89.6% |
| Latency mean / p50 / p95 | 1.203 / 0.979 / 2.494 s |
| Wall time | 609.3 s |
| Mean prompt tokens | 498.8 |
| Share of prompt tokens served from prefix cache | None |

| Level | EX strict, 95% CI |
|---|---|
| all | 73.0% [69.0, 77.0] (n=500) |

Errors: exec_error 52

95% intervals: percentile bootstrap over questions, 2,000 resamples, seed 0.
