# Run `E1_qwen15b_3shot_hf_olist-dev`

- Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` via `hf`; dataset `olist-dev`; prompt v1
- Definitions in prompt: False; few-shot examples: 3; sample rows: 2; concurrency 1; max_tokens 256
- Created 2026-11-25T07:43:06Z on Linux x86_64 py3.13.16; git None

| Metric | Value |
|---|---|
| **Execution accuracy (strict)** | **17.5% [7.5, 30.0] (n=40)** |
| Execution accuracy (relaxed, extra columns allowed) | 17.5% [7.5, 30.0] (n=40) |
| Valid SQL rate | 37.5% |
| Latency mean / p50 / p95 | 2.865 / 2.765 / 5.923 s |
| Wall time | 116.4 s |
| Mean prompt tokens | 2111.8 |
| Share of prompt tokens served from prefix cache | None |

| Level | EX strict, 95% CI |
|---|---|
| definition | 0.0% [0.0, 0.0] (n=5) |
| easy | 50.0% [20.0, 80.0] (n=10) |
| hard | 0.0% [0.0, 0.0] (n=12) |
| medium | 15.4% [0.0, 38.5] (n=13) |

Errors: exec_error 24, unsafe 1

95% intervals: percentile bootstrap over questions, 2,000 resamples, seed 0.
