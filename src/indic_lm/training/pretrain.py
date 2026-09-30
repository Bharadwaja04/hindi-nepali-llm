"""Pretraining loop for the decoder-only LM (next-token prediction).

Run::

    python -m indic_lm.training.pretrain \
        --language hindi \
        --tokenizer artifacts/tokenizers/hindi_bpe_31744.json \
        --manual-data  <manual.jsonl ...> \
        --nonmanual-data <nonmanual.jsonl ...> \
        --checkpoint-dir checkpoints/hindi --log logs/hindi_pretrain.jsonl

Differences from the original ``hindi/train/train.py`` (which could not run
as committed: it referenced an undefined ``jsonl_document_stream`` / ``args.data``
and loaded the tokenizer with the HuggingFace ``tokenizers`` package, which
cannot parse this project's own BPE file format):

* the tokenizer is the project's own ``indic_lm.tokenizer.Tokenizer``;
* one config-driven entry point serves both languages;
* ``--resume`` restores model/optimizer/scheduler/scaler and skips the batches
  already consumed, so a resumed run sees the same data order;
* optional validation loss every ``--eval-every`` steps.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import torch
import torch.nn as nn

from indic_lm.data.streams import batch_stream, combined_token_stream, jsonl_document_stream
from indic_lm.languages import get_language
from indic_lm.model.transformer import DecoderOnlyLM
from indic_lm.tokenizer import Tokenizer
from indic_lm.training.utils import (
    append_log,
    calculate_perplexity,
    create_optimizer,
    create_scheduler,
    get_learning_rate,
    load_checkpoint,
    save_checkpoint,
    set_warmup_lr,
)


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def evaluate_loss(model, batches, device, max_batches: int = 50) -> float:
    """Mean cross-entropy over up to ``max_batches`` validation batches."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, n = 0.0, 0
    for x, y in itertools.islice(batches, max_batches):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        n += 1
    model.train()
    return total / max(n, 1)


def train(
    model,
    batch_iterator,
    config: dict,
    device,
    checkpoint_dir,
    log_path,
    val_batches_fn=None,
    eval_every: int = 0,
    resume_from=None,
):
    lr = config["learning_rate"]
    max_steps = config["max_steps"]
    warmup_steps = config["warmup_steps"]
    accum = config["grad_accum_steps"]

    optimizer = create_optimizer(model, lr, config["weight_decay"])
    scheduler = create_scheduler(optimizer, max_steps, warmup_steps, lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()

    start_step, tokens_seen = 1, 0
    if resume_from:
        ckpt = load_checkpoint(resume_from, model, optimizer, scheduler, scaler)
        start_step = ckpt["step"] + 1
        tokens_seen = ckpt["tokens_seen"]
        # Deterministic stream: skip the micro-batches that were already used.
        for _ in itertools.islice(batch_iterator, (start_step - 1) * accum):
            pass
        print(f"Resumed from {resume_from} at step {ckpt['step']}")

    model.to(device).train()
    os.makedirs(checkpoint_dir, exist_ok=True)

    for step in range(start_step, max_steps + 1):
        step_loss = 0.0
        for _ in range(accum):
            try:
                x, y = next(batch_iterator)
            except StopIteration:
                print("Data stream exhausted; stopping.")
                return
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits.reshape(-1, config["vocab_size"]), y.reshape(-1)) / accum

            scaler.scale(loss).backward()
            step_loss += loss.item()
            tokens_seen += x.numel()

        # Linear warmup, then cosine decay to 10% of the peak learning rate.
        if step <= warmup_steps:
            set_warmup_lr(optimizer, lr, step, warmup_steps)

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step > warmup_steps:
            scheduler.step()

        if step % config["log_every"] == 0:
            record = {
                "step": step,
                "loss": step_loss,
                "perplexity": calculate_perplexity(step_loss),
                "tokens_seen": tokens_seen,
                "learning_rate": get_learning_rate(optimizer),
                "grad_norm": float(grad_norm),
            }
            if val_batches_fn is not None and eval_every and step % eval_every == 0:
                record["val_loss"] = evaluate_loss(model, val_batches_fn(), device)
                record["val_perplexity"] = calculate_perplexity(record["val_loss"])
            append_log(log_path, record)
            print(
                f"Step {step}/{max_steps} | loss {step_loss:.4f} | ppl {record['perplexity']:.2f} "
                f"| tokens {tokens_seen:,} | lr {record['learning_rate']:.2e}"
            )

        if step % config["save_every"] == 0:
            save_checkpoint(
                os.path.join(checkpoint_dir, f"checkpoint_step_{step:06d}.pt"),
                step, tokens_seen, model, optimizer, scheduler, scaler, config,
            )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pretrain a language-specific decoder-only LM.")
    p.add_argument("--language", required=True, choices=["hindi", "nepali"])
    p.add_argument("--config", help="Model/training config JSON (default: configs/<language>/model.json).")
    p.add_argument("--tokenizer", help="Tokenizer JSON (default: artifacts/tokenizers/<language>_bpe_31744.json).")
    p.add_argument("--manual-data", nargs="+", required=True, help="Manual-corpus JSONL files (all documents used).")
    p.add_argument("--nonmanual-data", nargs="+", required=True, help="Non-manual JSONL files (split=='train' used).")
    p.add_argument("--val-data", nargs="*", default=[], help="Optional JSONL files with split=='validation' records.")
    p.add_argument("--checkpoint-dir")
    p.add_argument("--log")
    p.add_argument("--resume", help="Checkpoint to resume from.")
    p.add_argument("--max-steps", type=int, help="Override config max_steps (smoke tests).")
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--seed", type=int, default=1337)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    spec = get_language(args.language)

    config = load_config(args.config or spec.model_config_path)
    if args.max_steps:
        config["max_steps"] = args.max_steps
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderOnlyLM.from_config(config)
    print(f"Device: {device} | parameters: {model.num_parameters():,}")

    tokenizer = Tokenizer.load(args.tokenizer or spec.tokenizer_path)

    tokens = combined_token_stream(
        manual_paths=args.manual_data,
        nonmanual_paths=args.nonmanual_data,
        tokenizer=tokenizer,
        manual_budget=config["manual_target_tokens"],
        nonmanual_budget=config["nonmanual_target_tokens"],
    )
    batches = batch_stream(tokens, batch_size=config["batch_size"], seq_len=config["seq_len"])

    val_fn = None
    if args.val_data:
        def val_fn():
            docs = jsonl_document_stream(args.val_data, split="validation")
            ids = (t for text in docs for t in tokenizer.encode(text))
            return batch_stream(ids, batch_size=config["batch_size"], seq_len=config["seq_len"])

    checkpoint_dir = args.checkpoint_dir or f"checkpoints/{args.language}"
    log_path = args.log or f"logs/{args.language}_pretrain.jsonl"
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    train(model, batches, config, device, checkpoint_dir, log_path,
          val_batches_fn=val_fn, eval_every=args.eval_every, resume_from=args.resume)


if __name__ == "__main__":
    main()
