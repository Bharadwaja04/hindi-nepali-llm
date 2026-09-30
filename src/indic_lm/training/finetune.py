"""Reasoning fine-tuning (continued next-token training on QA text).

Refactored from the Kaggle notebook cells in ``notebooks/03_phase3_finetune_kaggle.ipynb``
(the original ``finetuning_reasoning_*.py`` files were raw cell dumps with
hard-coded ``/kaggle/...`` paths). Hyper-parameters are the ones the notebook used:

    AdamW, lr 1e-5, weight decay 0.01, batch 4, 3 epochs, grad-clip 1.0,
    padding label -100 (ignored by the loss), keep the best-validation checkpoint.

Two text formats are supported:
    qa   -> "प्रश्न: {question}\n\nउत्तर: {answer}"                (answer only)
    full -> the record's "text" field: question + समाधान (reasoning) + उत्तर

    python -m indic_lm.training.finetune --language nepali \
        --base-checkpoint models/nepali_pretrained.pt \
        --train data/reasoning/nepali/splits/train.jsonl \
        --val   data/reasoning/nepali/splits/validation.jsonl --format full
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from indic_lm.evaluation.reasoning import load_model
from indic_lm.languages import LanguageSpec, get_language
from indic_lm.tokenizer import Tokenizer

IGNORE_INDEX = -100


def build_text(record: dict, spec: LanguageSpec, fmt: str) -> str:
    if fmt == "full" and record.get("text"):
        return record["text"]
    return f"{spec.question_label}: {record['question'].strip()}\n\n{spec.answer_label}: {record['answer'].strip()}"


def load_examples(path, spec: LanguageSpec, fmt: str) -> list[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not str(rec.get("question", "")).strip() or not str(rec.get("answer", "")).strip():
                continue
            texts.append(build_text(rec, spec, fmt))
    return texts


class QADataset(Dataset):
    """Right-padded (input, label) pairs; padded label positions are ignored."""

    def __init__(self, texts: list[str], tokenizer: Tokenizer, seq_len: int = 128, pad_id: int = 0):
        self.texts, self.tokenizer, self.seq_len, self.pad_id = texts, tokenizer, seq_len, pad_id

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        ids = self.tokenizer.encode(self.texts[idx])[: self.seq_len + 1]
        x, y = ids[:-1], ids[1:]
        pad = self.seq_len - len(x)
        if pad > 0:
            x = x + [self.pad_id] * pad
            y = y + [IGNORE_INDEX] * pad
        return {"input_ids": torch.tensor(x, dtype=torch.long), "labels": torch.tensor(y, dtype=torch.long)}


def masked_loss(logits: torch.Tensor, labels: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                           ignore_index=IGNORE_INDEX, reduction=reduction)


@torch.no_grad()
def evaluate_loader(model, loader, device) -> tuple[float, float]:
    """Token-weighted mean loss and perplexity (padding excluded)."""
    model.eval()
    total, tokens = 0.0, 0
    for batch in loader:
        x, y = batch["input_ids"].to(device), batch["labels"].to(device)
        total += masked_loss(model(x), y, reduction="sum").item()
        tokens += int((y != IGNORE_INDEX).sum())
    loss = total / max(tokens, 1)
    return loss, math.exp(min(loss, 20))


def finetune(model, train_loader, val_loader, device, epochs=3, lr=1e-5, weight_decay=0.01,
             grad_clip=1.0, output=None, max_steps=None) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "val_loss": [], "val_perplexity": []}
    best = float("inf")
    step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running, batches = 0.0, 0
        for batch in train_loader:
            x, y = batch["input_ids"].to(device), batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running += loss.item()
            batches += 1
            step += 1
            if max_steps and step >= max_steps:
                break

        val_loss, val_ppl = evaluate_loader(model, val_loader, device)
        history["train_loss"].append(running / max(batches, 1))
        history["val_loss"].append(val_loss)
        history["val_perplexity"].append(val_ppl)
        print(
            f"epoch {epoch}/{epochs} | train {history['train_loss'][-1]:.4f} "
            f"| val {val_loss:.4f} | ppl {val_ppl:.2f}"
        )

        if output and val_loss < best:
            best = val_loss
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "best_val_loss": best}, output)
        if max_steps and step >= max_steps:
            break
    return history


def main(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune a pretrained LM on reasoning QA.")
    p.add_argument("--language", required=True, choices=["hindi", "nepali"])
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--tokenizer")
    p.add_argument("--format", choices=["qa", "full"], default="qa")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--output", default="models/finetuned.pt")
    p.add_argument("--max-steps", type=int, help="Stop early (smoke tests).")
    args = p.parse_args(argv)

    spec = get_language(args.language)
    config = spec.load_model_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer or spec.tokenizer_path)

    model = load_model(args.base_checkpoint, {**config, "dropout": config.get("dropout", 0.1)}, device)
    train_ds = QADataset(load_examples(args.train, spec, args.format), tokenizer, config["seq_len"])
    val_ds = QADataset(load_examples(args.val, spec, args.format), tokenizer, config["seq_len"])
    print(f"train {len(train_ds)} | val {len(val_ds)} | format {args.format}")

    history = finetune(
        model,
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=args.batch_size),
        device, args.epochs, args.lr, args.weight_decay, output=args.output, max_steps=args.max_steps,
    )
    metrics_path = Path(args.output).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
