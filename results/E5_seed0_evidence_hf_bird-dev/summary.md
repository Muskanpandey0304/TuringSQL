# Run `E5_seed0_evidence_hf_bird-dev`

- Model: `/kaggle/working/turingsql-1.5b` via `hf`; dataset `bird-dev`; prompt v1
- Definitions in prompt: True; few-shot examples: 0; sample rows: 2; concurrency 1; max_tokens 256
- Created 2026-11-25T08:07:35Z on Linux x86_64 py3.13.16; git None

| Metric | Value |
|---|---|
| **Execution accuracy (strict)** | **34.0% [30.0, 38.2] (n=500)** |
| Execution accuracy (relaxed, extra columns allowed) | 34.2% [30.2, 38.4] (n=500) |
| Valid SQL rate | 65.2% |
| Latency mean / p50 / p95 | 2.359 / 2.23 / 3.986 s |
| Wall time | 1192.1 s |
| Mean prompt tokens | 1328.1 |
| Share of prompt tokens served from prefix cache | None |

| Level | EX strict, 95% CI |
|---|---|
| all | 34.0% [30.0, 38.2] (n=500) |

Errors: exec_error 162, unsafe 12

95% intervals: percentile bootstrap over questions, 2,000 resamples, seed 0.
