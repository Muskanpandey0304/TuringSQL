# Run `E1_qwen15b_3shot_hf_bird-dev`

- Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` via `hf`; dataset `bird-dev`; prompt v1
- Definitions in prompt: False; few-shot examples: 3; sample rows: 2; concurrency 1; max_tokens 256
- Created 2026-11-25T07:40:45Z on Linux x86_64 py3.13.16; git None

| Metric | Value |
|---|---|
| **Execution accuracy (strict)** | **16.6% [13.6, 19.8] (n=500)** |
| Execution accuracy (relaxed, extra columns allowed) | 16.6% [13.6, 19.8] (n=500) |
| Valid SQL rate | 54.4% |
| Latency mean / p50 / p95 | 2.276 / 2.187 / 3.955 s |
| Wall time | 1158.9 s |
| Mean prompt tokens | 1696.2 |
| Share of prompt tokens served from prefix cache | None |

| Level | EX strict, 95% CI |
|---|---|
| all | 16.6% [13.6, 19.8] (n=500) |

Errors: exec_error 214, too_many_rows 4, unsafe 9, timeout 1

95% intervals: percentile bootstrap over questions, 2,000 resamples, seed 0.
