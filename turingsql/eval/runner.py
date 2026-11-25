"""Run a model over a question set, score by execution, and write a reproducible report.

Every example is a dict with: id, db_path, question, gold_sql, order_matters, and optionally level,
definitions (list of strings). Olist, Spider and BIRD are all converted to this shape.

Engines:
- "openai": any OpenAI-compatible server (TuringServe, vLLM, Ollama's /v1). Requests run concurrently,
  which also exercises the server's continuous batching.
- "hf": transformers greedy decoding in-process (the reference everyone understands).

Decoding is always pure greedy with every setting passed explicitly (docs/DECISIONS.md, decision 3).
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from turingsql.eval.execution import run_sql, score
from turingsql.eval.stats import bootstrap_ci
from turingsql.fewshot import build_fewshot_messages
from turingsql.prompt import PROMPT_VERSION, build_messages, extract_sql
from turingsql.schema import serialize_schema


@dataclass
class Generation:
    text: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None


@dataclass
class RunConfig:
    run_id: str
    model: str
    engine: str
    dataset: str
    max_tokens: int = 256
    concurrency: int = 1
    n_sample_rows: int = 2
    use_definitions: bool = False
    n_shots: int = 0
    notes: str = ""
    extra: dict = field(default_factory=dict)


Generator = Callable[[list[dict[str, str]]], Generation]


# ---------- engines ----------

def openai_generator(base_url: str, model: str, max_tokens: int, api_key: str = "none",
                     timeout_s: float = 600.0) -> Generator:
    from openai import OpenAI

    url = base_url.rstrip("/")
    client = OpenAI(base_url=url if url.endswith("/v1") else url + "/v1", api_key=api_key, timeout=timeout_s)

    def generate(messages: list[dict[str, str]]) -> Generation:
        t0 = time.perf_counter()
        r = client.chat.completions.create(model=model, messages=messages,  # type: ignore[arg-type]
                                           max_tokens=max_tokens, temperature=0, top_p=1)
        latency = time.perf_counter() - t0
        u = r.usage
        cached = None
        if u is not None and getattr(u, "prompt_tokens_details", None) is not None:
            cached = getattr(u.prompt_tokens_details, "cached_tokens", None)
        return Generation(r.choices[0].message.content or "", latency,
                          u.prompt_tokens if u else None, u.completion_tokens if u else None, cached)

    return generate


def hf_generator(model_path: str, max_tokens: int, device: str = "cpu", dtype: str = "float32") -> Generator:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch_dtype)
    model = model.to(device).eval()  # type: ignore[arg-type]

    def generate(messages: list[dict[str, str]]) -> Generation:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_tokens,
                                 do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None,
                                 top_k=None, pad_token_id=tok.eos_token_id)
        latency = time.perf_counter() - t0
        new = out[0, ids.shape[1]:]
        return Generation(tok.decode(new, skip_special_tokens=True), latency, ids.shape[1], int(new.shape[0]))

    return generate


# ---------- running ----------

def _git_commit(path: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def run(examples: Sequence[dict], generate: Generator, cfg: RunConfig, out_dir: Path,
        progress: bool = True) -> dict:
    """Generate, score and write predictions.jsonl, summary.json and summary.md into out_dir/run_id."""
    run_dir = out_dir / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    schemas: dict[str, str] = {}
    gold_rows: dict[str, list] = {}
    for ex in examples:
        db = str(ex["db_path"])
        if db not in schemas:
            schemas[db] = serialize_schema(db, n_sample_rows=cfg.n_sample_rows)
        res = run_sql(db, ex["gold_sql"], timeout_s=120)
        if res.rows is None:
            raise RuntimeError(f"gold SQL failed for {ex['id']}: {res.error} {res.message}")
        gold_rows[ex["id"]] = res.rows

    def one(ex: dict) -> dict:
        defs = ex.get("definitions") if cfg.use_definitions else None
        schema = schemas[str(ex["db_path"])]
        if ex.get("shots"):
            messages = build_fewshot_messages(ex["shots"], schema, ex["question"], defs)
        else:
            messages = build_messages(schema, ex["question"], defs)
        gen_error: str | None = None
        try:
            gen = generate(messages)
        except Exception as e:  # noqa: BLE001 - a failed request is a wrong answer, not a crash
            gen = Generation("", 0.0)
            gen_error = f"{type(e).__name__}: {e}"
        s = score(ex["db_path"], ex["gold_sql"], gen.text, ex.get("order_matters", False),
                  gold_rows=gold_rows[ex["id"]])
        return {
            "id": ex["id"], "level": ex.get("level"), "db": Path(ex["db_path"]).stem,
            "question": ex["question"], "gold_sql": ex["gold_sql"],
            "raw_output": gen.text, "pred_sql": extract_sql(gen.text),
            "correct": s.correct, "correct_relaxed": s.correct_relaxed,
            "error": "generation_error" if gen_error else s.error, "message": gen_error or s.message,
            "latency_s": round(gen.latency_s, 3), "prompt_tokens": gen.prompt_tokens,
            "completion_tokens": gen.completion_tokens, "cached_tokens": gen.cached_tokens,
        }

    t0 = time.perf_counter()
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as pool:
        for i, rec in enumerate(pool.map(one, examples), 1):
            records.append(rec)
            if progress:
                mark = "ok " if rec["correct"] else (rec["error"] or "wrong")
                print(f"[{i:>4}/{len(examples)}] {rec['id']:<12} {mark:<14} {rec['latency_s']:>6.1f}s", flush=True)
    wall = time.perf_counter() - t0

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize(records, cfg, wall)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def _ci(values: list[float]) -> dict:
    mean, low, high = bootstrap_ci(values)
    return {"n": len(values), "acc": round(mean, 4), "ci95": [round(low, 4), round(high, 4)]}


def summarize(records: list[dict], cfg: RunConfig, wall_s: float) -> dict:
    by_level: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_level[r["level"] or "all"].append(float(r["correct"]))
    lat = sorted(r["latency_s"] for r in records)
    p_tok = [r["prompt_tokens"] for r in records if r["prompt_tokens"]]
    cached = [r["cached_tokens"] for r in records if r["cached_tokens"] is not None]
    root = Path(__file__).resolve().parents[2]
    return {
        "config": asdict(cfg),
        "prompt_version": PROMPT_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(root),
        "platform": f"{platform.system()} {platform.machine()} py{platform.python_version()}",
        "ex_strict": _ci([float(r["correct"]) for r in records]),
        "ex_relaxed": _ci([float(r["correct_relaxed"]) for r in records]),
        "valid_sql_rate": round(sum(r["error"] is None for r in records) / max(1, len(records)), 4),
        "by_level": {k: _ci(v) for k, v in sorted(by_level.items())},
        "errors": dict(Counter(r["error"] for r in records if r["error"])),
        "latency_s": {"mean": round(sum(lat) / max(1, len(lat)), 3),
                      "p50": lat[len(lat) // 2] if lat else None,
                      "p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None},
        "wall_s": round(wall_s, 1),
        "mean_prompt_tokens": round(sum(p_tok) / len(p_tok), 1) if p_tok else None,
        "cached_token_share": round(sum(cached) / sum(p_tok), 4) if cached and p_tok else None,
    }


def render_markdown(s: dict) -> str:
    c = s["config"]

    def pct(d: dict) -> str:
        return f"{100 * d['acc']:.1f}% [{100 * d['ci95'][0]:.1f}, {100 * d['ci95'][1]:.1f}] (n={d['n']})"

    lines = [
        f"# Run `{c['run_id']}`", "",
        f"- Model: `{c['model']}` via `{c['engine']}`; dataset `{c['dataset']}`; prompt {s['prompt_version']}",
        f"- Definitions in prompt: {c['use_definitions']}; few-shot examples: {c.get('n_shots', 0)}; "
        f"sample rows: {c['n_sample_rows']}; "
        f"concurrency {c['concurrency']}; max_tokens {c['max_tokens']}",
        f"- Created {s['created_utc']} on {s['platform']}; git {s['git_commit']}",
    ]
    if c.get("notes"):
        lines.append(f"- Notes: {c['notes']}")
    lines += [
        "", "| Metric | Value |", "|---|---|",
        f"| **Execution accuracy (strict)** | **{pct(s['ex_strict'])}** |",
        f"| Execution accuracy (relaxed, extra columns allowed) | {pct(s['ex_relaxed'])} |",
        f"| Valid SQL rate | {100 * s['valid_sql_rate']:.1f}% |",
        f"| Latency mean / p50 / p95 | {s['latency_s']['mean']} / {s['latency_s']['p50']} / "
        f"{s['latency_s']['p95']} s |",
        f"| Wall time | {s['wall_s']} s |",
        f"| Mean prompt tokens | {s['mean_prompt_tokens']} |",
        f"| Share of prompt tokens served from prefix cache | {s['cached_token_share']} |",
        "", "| Level | EX strict, 95% CI |", "|---|---|",
        *[f"| {k} | {pct(v)} |" for k, v in s["by_level"].items()],
        "", "Errors: " + (", ".join(f"{k} {v}" for k, v in s["errors"].items()) or "none"), "",
        "95% intervals: percentile bootstrap over questions, 2,000 resamples, seed 0.", "",
    ]
    return "\n".join(lines)
