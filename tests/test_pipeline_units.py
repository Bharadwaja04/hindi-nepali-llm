import math

import pytest
import torch

from indic_lm.data.streams import batch_stream, budgeted_token_stream, combined_token_stream
from indic_lm.evaluation.metrics import (
    calculate_bits_per_token,
    calculate_bpb,
    calculate_perplexity,
    distinct_n,
    repetition_rate,
)
from indic_lm.evaluation.reasoning import (
    baselines,
    extract_answer,
    is_exact,
    is_lenient,
    normalize_answer,
)
from indic_lm.training.utils import (
    create_optimizer,
    create_scheduler,
    load_checkpoint,
    save_checkpoint,
    set_warmup_lr,
)


# ---- metrics: values checked against closed forms -------------------------------------------
def test_perplexity_and_bits_per_token():
    assert calculate_perplexity(0.0) == 1.0
    assert calculate_perplexity(2.0) == pytest.approx(math.e**2)
    assert calculate_bits_per_token(math.log(2)) == pytest.approx(1.0)
    assert calculate_bpb(math.log(2) * 4, average_bytes_per_token=4) == pytest.approx(1.0)


def test_reported_hindi_metrics_are_self_consistent():
    """Phase 2 report: loss 2.549785 -> ppl 12.8044, 3.678562 bits/token."""
    assert calculate_perplexity(2.549785) == pytest.approx(12.8044, abs=1e-3)
    assert calculate_bits_per_token(2.549785) == pytest.approx(3.678562, abs=1e-5)


def test_distinct_n_and_repetition():
    seq = [1, 2, 3, 1, 2, 3]
    assert distinct_n(seq, 1) == pytest.approx(3 / 6)
    assert distinct_n(seq, 2) == pytest.approx(3 / 5)
    assert repetition_rate([1, 2, 3, 4, 5], n=3) == 0.0
    assert repetition_rate([7] * 10, n=3) == pytest.approx(1 - 1 / 8)
    assert distinct_n([1], 2) == 0.0


# ---- token streams --------------------------------------------------------------------------
def test_batch_stream_shapes_and_next_token_alignment():
    x, y = next(batch_stream(iter(range(1000)), batch_size=2, seq_len=4))
    assert x.shape == y.shape == (2, 4)
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(y.flatten()[:-1], x.flatten()[1:])  # contiguous stream


def test_batch_stream_carries_boundary_token_between_batches():
    it = batch_stream(iter(range(100)), batch_size=2, seq_len=4)
    x1, y1 = next(it)
    x2, _ = next(it)
    assert x2[0, 0].item() == y1[-1, -1].item()  # last target of batch 1 starts batch 2


def test_batch_stream_drops_incomplete_final_batch():
    batches = list(batch_stream(iter(range(1 + 2 * 4 * 3 + 5)), batch_size=2, seq_len=4))
    assert len(batches) == 3


def test_budgeted_token_stream_respects_budget(tiny_tokenizer):
    docs = ["भारत एक सुंदर देश है।"] * 50
    out = list(budgeted_token_stream(iter(docs), tiny_tokenizer, token_budget=100))
    assert len(out) == 100


def test_combined_stream_uses_only_train_split_of_nonmanual(tiny_tokenizer, tmp_path):
    import json

    manual = tmp_path / "m.jsonl"
    non = tmp_path / "n.jsonl"
    manual.write_text(json.dumps({"text": "भारत"}, ensure_ascii=False) + "\n", encoding="utf-8")
    non.write_text(
        json.dumps({"text": "नेपाल", "split": "train"}, ensure_ascii=False) + "\n"
        + json.dumps({"text": "दिल्ली", "split": "test"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    got = list(combined_token_stream([manual], [non], tiny_tokenizer, 1000, 1000))
    expected = tiny_tokenizer.encode("भारत") + tiny_tokenizer.encode("नेपाल")
    assert got == expected


# ---- optimisation utilities -----------------------------------------------------------------
def test_warmup_then_cosine_to_ten_percent():
    model = torch.nn.Linear(2, 2)
    lr, warmup, total = 1e-3, 5, 25
    opt = create_optimizer(model, lr, 0.1)
    sched = create_scheduler(opt, total, warmup, lr)

    for step in range(1, warmup + 1):
        set_warmup_lr(opt, lr, step, warmup)
        assert opt.param_groups[0]["lr"] == pytest.approx(lr * step / warmup)
    for _ in range(total - warmup):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1 * lr, rel=1e-6)


def test_checkpoint_round_trip(tmp_path):
    model = torch.nn.Linear(4, 4)
    opt = create_optimizer(model, 1e-3, 0.1)
    sched = create_scheduler(opt, 10, 2, 1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = tmp_path / "ck.pt"
    save_checkpoint(path, 7, 1234, model, opt, sched, scaler, {"a": 1})

    other = torch.nn.Linear(4, 4)
    opt2 = create_optimizer(other, 1e-3, 0.1)
    ck = load_checkpoint(path, other, opt2, create_scheduler(opt2, 10, 2, 1e-3), scaler)
    assert ck["step"] == 7 and ck["tokens_seen"] == 1234
    assert torch.equal(model.weight, other.weight)


# ---- reasoning-eval helpers -----------------------------------------------------------------
def test_answer_extraction_and_matching():
    assert extract_answer(" अजय\nप्रश्न: ...", "उत्तर") == "अजय"
    assert extract_answer("समाधान: 90 > 50।\n\nउत्तर: कविता\n", "उत्तर") == "कविता"
    assert normalize_answer(" कविता। ") == "कविता"
    assert is_exact("कविता।", "कविता")
    assert not is_exact("कविता और मीना", "कविता")
    assert is_lenient("कविता की वजन अधिक है", "कविता")
    assert not is_lenient("", "")


def test_baselines_on_toy_records():
    recs = [
        {"question": "क बड़ा है या ख?", "answer": "क"},
        {"question": "क बड़ा है या ख?", "answer": "क"},
        {"question": "ग बड़ा है या ख?", "answer": "ख"},
    ]
    b = baselines(recs)
    assert b["majority_answer"] == "क"
    assert b["majority_exact_match"] == pytest.approx(2 / 3)
    assert 0 < b["informed_chance_exact_match"] <= 1
