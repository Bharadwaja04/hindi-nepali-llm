"""Shared fixtures: a tiny trained tokenizer and a tiny model config, so tests run in seconds."""

import json

import pytest

from indic_lm.tokenizer import Tokenizer

HINDI_TEXT = (
    "भारत एक सुंदर देश है। दिल्ली भारत की राजधानी है। "
    "शिक्षा के क्षेत्र में तकनीक का उपयोग बढ़ रहा है। "
) * 40

NEPALI_TEXT = (
    "नेपाल सुन्दर देश हो। काठमाडौँ नेपालको राजधानी हो। "
    "कृषि क्षेत्रको विकास आवश्यक छ। "
) * 40


@pytest.fixture(scope="session")
def tiny_tokenizer() -> Tokenizer:
    tok = Tokenizer("hindi", vocab_size=320)
    tok.train(HINDI_TEXT + NEPALI_TEXT, verbose=False)
    return tok


@pytest.fixture()
def tiny_config() -> dict:
    return {
        "language": "hindi",
        "vocab_size": 320,
        "d_model": 32,
        "num_heads": 4,
        "num_layers": 2,
        "d_ff": 64,
        "seq_len": 16,
        "dropout": 0.0,
        "batch_size": 2,
        "grad_accum_steps": 2,
        "learning_rate": 1e-3,
        "weight_decay": 0.1,
        "warmup_steps": 2,
        "max_grad_norm": 1.0,
        "max_steps": 6,
        "log_every": 1,
        "save_every": 3,
        "manual_target_tokens": 4000,
        "nonmanual_target_tokens": 4000,
    }


@pytest.fixture()
def tiny_corpus(tmp_path):
    """Manual + non-manual JSONL files in the format the token streams expect."""
    manual = tmp_path / "manual.jsonl"
    nonmanual = tmp_path / "nonmanual.jsonl"
    with manual.open("w", encoding="utf-8") as f:
        for i in range(30):
            f.write(json.dumps({"text": HINDI_TEXT[i * 20 : i * 20 + 300]}, ensure_ascii=False) + "\n")
    with nonmanual.open("w", encoding="utf-8") as f:
        for i in range(30):
            f.write(json.dumps({"text": NEPALI_TEXT[i * 20 : i * 20 + 300], "split": "train"},
                               ensure_ascii=False) + "\n")
    return manual, nonmanual
