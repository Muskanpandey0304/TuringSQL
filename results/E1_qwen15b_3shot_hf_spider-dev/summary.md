# Run `E1_qwen15b_3shot_hf_spider-dev`

- Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` via `hf`; dataset `spider-dev`; prompt v1
- Definitions in prompt: False; few-shot examples: 3; sample rows: 2; concurrency 1; max_tokens 256
- Created 2026-11-25T07:19:10Z on Linux x86_64 py3.13.16; git None

| Metric | Value |
|---|---|
| **Execution accuracy (strict)** | **63.2% [59.0, 67.4] (n=500)** |
| Execution accuracy (relaxed, extra columns allowed) | 63.4% [59.4, 67.6] (n=500) |
| Valid SQL rate | 82.0% |
| Latency mean / p50 / p95 | 1.32 / 1.019 / 2.723 s |
| Wall time | 668.3 s |
| Mean prompt tokens | 863.9 |
| Share of prompt tokens served from prefix cache | None |

| Level | EX strict, 95% CI |
|---|---|
| all | 63.2% [59.0, 67.4] (n=500) |

Errors: exec_error 90

95% intervals: percentile bootstrap over questions, 2,000 resamples, seed 0.
