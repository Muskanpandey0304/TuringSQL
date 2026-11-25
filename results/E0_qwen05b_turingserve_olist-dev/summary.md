# Run `E0_qwen05b_turingserve_olist-dev`

- Model: `qwen-coder-0.5b` via `openai`; dataset `olist-dev`; prompt v1
- Definitions in prompt: False; sample rows: 2; concurrency 4; max_tokens 256
- Created 2026-11-24T07:44:54Z on Windows AMD64 py3.10.11; git None
- Notes: Base Qwen2.5-Coder-0.5B-Instruct, zero-shot, TuringServe CPU reference kernels, fp32

| Metric | Value |
|---|---|
| **Execution accuracy (strict)** | **22.5% [10.0, 37.5] (n=40)** |
| Execution accuracy (relaxed, extra columns allowed) | 22.5% [10.0, 37.5] (n=40) |
| Valid SQL rate | 47.5% |
| Latency mean / p50 / p95 | 31.432 / 32.903 / 63.21 s |
| Wall time | 326.0 s |
| Mean prompt tokens | 1716.6 |
| Share of prompt tokens served from prefix cache | 0.9542 |

| Level | EX strict, 95% CI |
|---|---|
| definition | 0.0% [0.0, 0.0] (n=5) |
| easy | 70.0% [40.0, 100.0] (n=10) |
| hard | 0.0% [0.0, 0.0] (n=12) |
| medium | 15.4% [0.0, 38.5] (n=13) |

Errors: exec_error 19, too_many_rows 1, unsafe 1

95% intervals: percentile bootstrap over questions, 2,000 resamples, seed 0.
