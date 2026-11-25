# Run `E0_qwen15b_hf_olist-dev`

- Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` via `hf`; dataset `olist-dev`; prompt v1
- Definitions in prompt: False; few-shot examples: 0; sample rows: 2; concurrency 1; max_tokens 256
- Created 2026-11-25T07:07:29Z on Linux x86_64 py3.13.16; git None

| Metric | Value |
|---|---|
| **Execution accuracy (strict)** | **32.5% [17.5, 47.5] (n=40)** |
| Execution accuracy (relaxed, extra columns allowed) | 32.5% [17.5, 47.5] (n=40) |
| Valid SQL rate | 60.0% |
| Latency mean / p50 / p95 | 3.129 / 2.83 / 9.03 s |
| Wall time | 128.7 s |
| Mean prompt tokens | 1716.6 |
| Share of prompt tokens served from prefix cache | None |

| Level | EX strict, 95% CI |
|---|---|
| definition | 0.0% [0.0, 0.0] (n=5) |
| easy | 70.0% [40.0, 90.0] (n=10) |
| hard | 8.3% [0.0, 25.0] (n=12) |
| medium | 38.5% [15.4, 61.5] (n=13) |

Errors: exec_error 16

95% intervals: percentile bootstrap over questions, 2,000 resamples, seed 0.
