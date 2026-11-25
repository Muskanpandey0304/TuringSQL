"""QLoRA supervised fine-tuning for text-to-SQL (PROJECT_PLAN.md section 6.1).

Reads chat-format JSONL built by data_prep/build_sft.py: one {"id", "messages": [system, user, assistant]}
per line, with the schema already inside the user turn, so training needs no databases.

Loss is computed on the assistant (SQL) tokens only. Plain transformers Trainer + peft, no TRL, so the
masking is explicit and testable.

Kaggle T4 (fp16, 4-bit NF4):
    python scripts/train.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --train data/processed/sft_train.jsonl \
        --val data/processed/sft_val.jsonl --out checkpoints/E3_seed0 --epochs 2 --seed 0 --report-to wandb

CPU smoke test (no quantisation, a few steps, tiny model):
    python scripts/train.py --model models/Qwen2.5-Coder-0.5B-Instruct --train data/processed/sft_train.jsonl \
        --val data/processed/sft_val.jsonl --out checkpoints/smoke --cpu-smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IGNORE = -100


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def tokenize_example(tok, messages: list[dict], max_len: int) -> dict | None:
    """input_ids for the whole chat and labels that are IGNORE everywhere except the assistant reply.

    Returns None if the example is longer than max_len (dropped, never truncated: a cut schema or a cut
    SQL answer would teach the wrong thing).
    """
    prompt_ids = tok.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True)
    full_ids = tok.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template: prompt is not a prefix of the full conversation")
    # Qwen's template ends the assistant turn with "<|im_end|>\n"; train on <|im_end|> (so the model learns
    # to stop) but not on the trailing newline.
    end = len(full_ids)
    if tok.decode(full_ids[-1:]) == "\n":
        end -= 1
    full_ids = full_ids[:end]
    if len(full_ids) > max_len:
        return None
    labels = [IGNORE] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "labels": labels}


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), IGNORE, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            labels[i, :k] = torch.tensor(b["labels"])
            mask[i, :k] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def build_dataset(tok, rows: list[dict], max_len: int) -> tuple[list[dict], int]:
    out, dropped = [], 0
    for r in rows:
        ex = tokenize_example(tok, r["messages"], max_len)
        if ex is None:
            dropped += 1
        else:
            out.append(ex)
    return out, dropped


def last_checkpoint(out_dir: Path) -> str | None:
    cks = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(cks[-1]) if cks else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--save-steps", type=int, default=250)
    ap.add_argument("--eval-steps", type=int, default=250)
    ap.add_argument("--val-limit", type=int, default=500, help="validation examples used for eval loss")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--push-adapter-to", default=None, help="optional HF repo id; adapters pushed at every save")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="save a checkpoint and stop cleanly after this many hours (Kaggle sessions end at 12 h)")
    ap.add_argument("--resume-from", type=Path, default=None,
                    help="folder holding an earlier run's checkpoint-* dirs (e.g. a previous Kaggle version's output)")
    ap.add_argument("--cpu-smoke", action="store_true", help="tiny CPU run: no 4-bit, 3 steps, 16 examples")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

    set_seed(args.seed)
    random.seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    limit = 16 if args.cpu_smoke else None
    train_rows = read_jsonl(args.train, limit)
    val_rows = read_jsonl(args.val, 8 if args.cpu_smoke else args.val_limit)
    t0 = time.time()
    train_ds, drop_t = build_dataset(tok, train_rows, args.max_len)
    val_ds, drop_v = build_dataset(tok, val_rows, args.max_len)
    n_label = sum(sum(x != IGNORE for x in e["labels"]) for e in train_ds)
    n_tok = sum(len(e["input_ids"]) for e in train_ds)
    print(f"train {len(train_ds)} (dropped {drop_t} > {args.max_len} tokens), val {len(val_ds)} (dropped {drop_v}); "
          f"{n_tok:,} tokens, {n_label:,} trained-on tokens; tokenised in {time.time() - t0:.0f}s", flush=True)

    if args.cpu_smoke:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    else:
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.float16)          # T4: no bf16
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, dtype=torch.float16,
                                                     device_map={"": 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(train_ds) / (args.batch * args.grad_accum))
    targs = TrainingArguments(
        output_dir=str(args.out),
        run_name=args.run_name or args.out.name,
        num_train_epochs=args.epochs,
        max_steps=3 if args.cpu_smoke else args.max_steps,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=1 if args.cpu_smoke else args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        optim="adamw_torch" if args.cpu_smoke else "paged_adamw_8bit",
        fp16=not args.cpu_smoke,
        bf16=False,
        gradient_checkpointing=not args.cpu_smoke,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1 if args.cpu_smoke else 10,
        eval_strategy="steps",
        eval_steps=2 if args.cpu_smoke else args.eval_steps,
        save_strategy="steps",
        save_steps=2 if args.cpu_smoke else args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=args.report_to,
        seed=args.seed,
        dataloader_num_workers=0,
        group_by_length=True,
        remove_unused_columns=False,
        use_cpu=args.cpu_smoke,
        push_to_hub=bool(args.push_adapter_to),
        hub_model_id=args.push_adapter_to,
        hub_strategy="every_save",
        hub_private_repo=True,
    )

    class StepTimer(TrainerCallback):
        def __init__(self) -> None:
            self.t = time.time()
            self.start_step = 0

        def on_train_begin(self, a, state, control, **kw):
            self.t, self.start_step = time.time(), state.global_step     # resumed runs start mid-way

        def on_log(self, a, state, control, logs=None, **kw):
            done = state.global_step - self.start_step
            if done > 0 and state.max_steps:
                per = (time.time() - self.t) / done
                eta = per * (state.max_steps - state.global_step)
                print(f"step {state.global_step}/{state.max_steps}  {per:.1f}s/step  "
                      f"eta {eta / 60:.0f} min  {logs}", flush=True)

    class TimeLimit(TrainerCallback):
        """Save and stop once --max-hours have passed, so the checkpoint lands in the session's output."""

        def __init__(self, hours: float | None) -> None:
            self.deadline = time.time() + hours * 3600 if hours else None
            self.hit = False

        def on_step_end(self, a, state, control, **kw):
            if self.deadline and time.time() > self.deadline:
                self.hit = True
                control.should_save = True
                control.should_training_stop = True
            return control

    timer = TimeLimit(args.max_hours)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=Collator(tok.pad_token_id), callbacks=[StepTimer(), timer])
    print(f"steps/epoch {steps_per_epoch}; effective batch {args.batch * args.grad_accum}", flush=True)
    if args.resume_from and last_checkpoint(args.out) is None:
        found = sorted(args.resume_from.rglob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        found = [p for p in found if p.is_dir() and (p / "trainer_state.json").is_file()]
        if found:
            # copy every checkpoint (the best one may not be the latest), read-only inputs cannot be written to
            args.out.mkdir(parents=True, exist_ok=True)
            for p in found:
                shutil.copytree(p, args.out / p.name, dirs_exist_ok=True)
            print(f"copied {len(found)} checkpoint(s) from {args.resume_from}", flush=True)
        else:
            print(f"no checkpoint found under {args.resume_from}; starting fresh", flush=True)
    resume = last_checkpoint(args.out)
    if resume:
        print(f"resuming from {resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)

    meta_base = {"base_model": args.model, "args": {k: str(v) for k, v in vars(args).items()},
                 "train_examples": len(train_ds), "val_examples": len(val_ds),
                 "global_step": trainer.state.global_step, "max_steps": trainer.state.max_steps,
                 "best_checkpoint": trainer.state.best_model_checkpoint, "best_eval_loss": trainer.state.best_metric,
                 "log_history": trainer.state.log_history}
    if timer.hit and trainer.state.global_step < trainer.state.max_steps:
        (args.out / "train_meta.json").write_text(json.dumps({**meta_base, "finished": False}, indent=2),
                                                  encoding="utf-8")
        print(f"TIME LIMIT: stopped at step {trainer.state.global_step}/{trainer.state.max_steps}; checkpoint saved. "
              "Run again with --resume-from pointing at this output to continue.", flush=True)
        return 3

    final = args.out / "final_adapter"
    trainer.model.save_pretrained(final)
    tok.save_pretrained(final)
    (args.out / "train_meta.json").write_text(json.dumps({**meta_base, "finished": True}, indent=2),
                                              encoding="utf-8")
    print(f"saved adapter to {final}; best eval loss {trainer.state.best_metric}", flush=True)
    if args.push_adapter_to:
        trainer.push_to_hub(commit_message="final adapter")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
