# Run `E0_qwen15b_turingserve_olist-dev`

- Model: `qwen-coder-1.5b` via `openai`; dataset `olist-dev`; prompt v1
- Definitions in prompt: False; sample rows: 2; concurrency 4; max_tokens 256
- Created 2026-11-24T08:24:02Z on Windows AMD64 py3.10.11; git None
- Notes: Base Qwen2.5-Coder-1.5B-Instruct, zero-shot, TuringServe CPU reference kernels, fp32

| Metric | Value |
|---|---|
| **Execution accuracy (strict)** | **32.5% [17.5, 47.5] (n=40)** |
| Execution accuracy (relaxed, extra columns allowed) | 32.5% [17.5, 47.5] (n=40) |
| Valid SQL rate | 60.0% |
| Latency mean / p50 / p95 | 89.312 / 73.131 / 258.873 s |
| Wall time | 913.9 s |
| Mean prompt tokens | 1716.6 |
| Share of prompt tokens served from prefix cache | 0.9542 |

| Level | EX strict, 95% CI |
|---|---|
| definition | 0.0% [0.0, 0.0] (n=5) |
| easy | 70.0% [40.0, 90.0] (n=10) |
| hard | 8.3% [0.0, 25.0] (n=12) |
| medium | 38.5% [15.4, 61.5] (n=13) |

Errors: exec_error 16

95% intervals: percentile bootstrap over questions, 2,000 resamples, seed 0.
