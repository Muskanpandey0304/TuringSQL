"""Merge a LoRA adapter into its base model and verify the merge (PROJECT_PLAN.md section 6.2).

TuringServe has no LoRA support, so it serves the merged weights. The merge is verified by greedy
decoding the same prompts with (base + adapter) and with the merged model: outputs must be identical.

    python scripts/merge.py --adapter checkpoints/E3_seed0/final_adapter --out models/turingsql-1.5b \
        --check-prompts data/processed/sft_val.jsonl --n-check 50 --dtype float16 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def greedy(model, tok, messages, device, max_new_tokens=128) -> str:
    import torch

    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                             do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default=None, help="defaults to the base model recorded in the adapter config")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--check-prompts", type=Path, default=None)
    ap.add_argument("--n-check", type=int, default=50)
    args = ap.parse_args()

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_id = args.base or PeftConfig.from_pretrained(str(args.adapter)).base_model_name_or_path
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(str(args.adapter))
    base = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype).to(args.device)
    peft_model = PeftModel.from_pretrained(base, str(args.adapter)).eval()

    prompts: list[list[dict]] = []
    if args.check_prompts:
        with args.check_prompts.open(encoding="utf-8") as f:
            for line in f:
                prompts.append(json.loads(line)["messages"][:-1])
                if len(prompts) >= args.n_check:
                    break
    before = [greedy(peft_model, tok, m, args.device) for m in prompts]

    merged = peft_model.merge_and_unload().eval()
    after = [greedy(merged, tok, m, args.device) for m in prompts]
    same = sum(a == b for a, b in zip(before, after, strict=True))
    print(f"merge check: {same}/{len(prompts)} greedy outputs identical")

    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    # Serve with plain greedy defaults; Qwen's shipped generation_config adds repetition_penalty 1.05.
    gen_cfg = {"bos_token_id": tok.bos_token_id, "eos_token_id": [tok.convert_tokens_to_ids("<|im_end|>"),
                                                                    tok.eos_token_id], "do_sample": False}
    (args.out / "generation_config.json").write_text(json.dumps(gen_cfg, indent=2), encoding="utf-8")
    (args.out / "merge_report.json").write_text(json.dumps(
        {"base": base_id, "adapter": str(args.adapter), "dtype": args.dtype, "checked": len(prompts),
         "identical": same}, indent=2), encoding="utf-8")
    print(f"saved merged model to {args.out}")
    if prompts and same < len(prompts):
        # fp16 merges can flip a near-tie token; report it loudly but keep the files for inspection.
        print("WARNING: some outputs differ after merging; inspect before serving", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
