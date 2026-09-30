"""End-to-end smoke tests: tiny model, tiny corpus, a few CPU steps."""

import itertools
import json

import torch

from indic_lm.data.streams import batch_stream, combined_token_stream
from indic_lm.model import DecoderOnlyLM
from indic_lm.training.finetune import IGNORE_INDEX, QADataset, finetune, masked_loss
from indic_lm.training.pretrain import train


def _batches(tokenizer, corpus, config):
    manual, nonmanual = corpus
    stream = combined_token_stream([manual], [nonmanual], tokenizer,
                                   config["manual_target_tokens"], config["nonmanual_target_tokens"])
    return batch_stream(stream, batch_size=config["batch_size"], seq_len=config["seq_len"])


def test_pretraining_runs_logs_and_checkpoints(tiny_tokenizer, tiny_config, tiny_corpus, tmp_path):
    torch.manual_seed(0)
    model = DecoderOnlyLM.from_config(tiny_config)
    log = tmp_path / "log.jsonl"
    train(model, _batches(tiny_tokenizer, tiny_corpus, tiny_config), tiny_config,
          torch.device("cpu"), tmp_path / "ck", log)

    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["step"] for r in rows] == list(range(1, tiny_config["max_steps"] + 1))
    assert all(r["loss"] == r["loss"] and r["loss"] > 0 for r in rows)  # finite, positive
    assert rows[-1]["tokens_seen"] == tiny_config["max_steps"] * tiny_config["grad_accum_steps"] \
        * tiny_config["batch_size"] * tiny_config["seq_len"]
    assert (tmp_path / "ck" / "checkpoint_step_000003.pt").exists()
    assert (tmp_path / "ck" / "checkpoint_step_000006.pt").exists()


def test_resume_reproduces_uninterrupted_training(tiny_tokenizer, tiny_config, tiny_corpus, tmp_path):
    """Steps 1-3, resume, steps 4-6 must equal steps 1-6 (dropout=0 -> deterministic)."""
    cpu = torch.device("cpu")

    torch.manual_seed(0)
    straight = DecoderOnlyLM.from_config(tiny_config)
    train(straight, _batches(tiny_tokenizer, tiny_corpus, tiny_config), tiny_config, cpu,
          tmp_path / "a", tmp_path / "a.jsonl")

    # Interrupted run: same schedule (max_steps=6) but the data runs out after 3 optimiser steps.
    torch.manual_seed(0)
    first = DecoderOnlyLM.from_config(tiny_config)
    limited = itertools.islice(_batches(tiny_tokenizer, tiny_corpus, tiny_config),
                               3 * tiny_config["grad_accum_steps"])
    train(first, limited, tiny_config, cpu, tmp_path / "b", tmp_path / "b1.jsonl")

    torch.manual_seed(123)  # different init on purpose: resume must overwrite it
    resumed = DecoderOnlyLM.from_config(tiny_config)
    train(resumed, _batches(tiny_tokenizer, tiny_corpus, tiny_config), tiny_config, cpu,
          tmp_path / "b", tmp_path / "b2.jsonl", resume_from=tmp_path / "b" / "checkpoint_step_000003.pt")

    for (n1, p1), (_, p2) in zip(straight.state_dict().items(), resumed.state_dict().items()):
        assert torch.allclose(p1, p2, atol=1e-6), n1


def test_finetune_masks_padding_and_updates_weights(tiny_tokenizer, tiny_config, tmp_path):
    texts = ["प्रश्न: भारत की राजधानी?\n\nउत्तर: दिल्ली", "प्रश्न: नेपाल की राजधानी?\n\nउत्तर: काठमाडौँ"]

    # Padding: with a context longer than the text, trailing labels are ignored by the loss.
    padded = QADataset(texts, tiny_tokenizer, seq_len=64)[0]
    assert padded["input_ids"].shape == padded["labels"].shape == (64,)
    assert (padded["labels"] == IGNORE_INDEX).any()
    assert int((padded["labels"] != IGNORE_INDEX).sum()) > 0

    # Truncation: with a short context there is no padding and exactly seq_len targets.
    ds = QADataset(texts, tiny_tokenizer, seq_len=tiny_config["seq_len"])
    item = ds[0]
    assert item["input_ids"].shape == item["labels"].shape == (tiny_config["seq_len"],)
    assert not (item["labels"] == IGNORE_INDEX).any()
    logits = torch.randn(1, tiny_config["seq_len"], tiny_config["vocab_size"])
    padded_only = torch.full((1, tiny_config["seq_len"]), IGNORE_INDEX)
    assert masked_loss(logits, item["labels"].unsqueeze(0)).isfinite()
    assert masked_loss(logits, padded_only, reduction="sum").item() == 0.0

    torch.manual_seed(0)
    model = DecoderOnlyLM.from_config(tiny_config)
    before = model.lm_head.weight.clone()
    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    hist = finetune(model, loader, loader, torch.device("cpu"), epochs=2, lr=1e-2, output=tmp_path / "ft.pt")
    assert not torch.equal(before, model.lm_head.weight)
    assert len(hist["val_loss"]) == 2 and (tmp_path / "ft.pt").exists()
