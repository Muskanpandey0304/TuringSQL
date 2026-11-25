# Run `E0_qwen15b_hf_spider-dev`

- Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` via `hf`; dataset `spider-dev`; prompt v1
- Definitions in prompt: False; few-shot examples: 0; sample rows: 2; concurrency 1; max_tokens 256
- Created 2026-11-24T20:36:50Z on Linux x86_64 py3.13.16; git None

| Metric | Value |
|---|---|
| **Execution accuracy (strict)** | **64.4% [60.0, 68.8] (n=500)** |
| Execution accuracy (relaxed, extra columns allowed) | 64.8% [60.6, 69.0] (n=500) |
| Valid SQL rate | 82.6% |
| Latency mean / p50 / p95 | 1.392 / 1.155 / 2.925 s |
| Wall time | 708.2 s |
| Mean prompt tokens | 498.8 |
| Share of prompt tokens served from prefix cache | None |

| Level | EX strict, 95% CI |
|---|---|
| all | 64.4% [60.0, 68.8] (n=500) |

Errors: exec_error 86, too_many_rows 1

95% intervals: percentile bootstrap over questions, 2,000 resamples, seed 0.
