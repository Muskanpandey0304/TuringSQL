# Run `E0_qwen15b_hf_bird-dev`

- Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` via `hf`; dataset `bird-dev`; prompt v1
- Definitions in prompt: False; few-shot examples: 0; sample rows: 2; concurrency 1; max_tokens 256
- Created 2026-11-25T07:04:57Z on Linux x86_64 py3.13.16; git None

| Metric | Value |
|---|---|
| **Execution accuracy (strict)** | **18.4% [15.0, 21.8] (n=500)** |
| Execution accuracy (relaxed, extra columns allowed) | 18.6% [15.2, 22.0] (n=500) |
| Valid SQL rate | 60.6% |
| Latency mean / p50 / p95 | 2.191 / 2.135 / 3.925 s |
| Wall time | 1117.6 s |
| Mean prompt tokens | 1299.9 |
| Share of prompt tokens served from prefix cache | None |

| Level | EX strict, 95% CI |
|---|---|
| all | 18.4% [15.0, 21.8] (n=500) |

Errors: unsafe 29, exec_error 166, timeout 1, too_many_rows 1

95% intervals: percentile bootstrap over questions, 2,000 resamples, seed 0.
