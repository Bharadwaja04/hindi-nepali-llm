import pytest
import torch

from indic_lm.languages import LANGUAGES
from indic_lm.model import DecoderOnlyLM, load_state_dict_compat


@pytest.fixture()
def tiny_model(tiny_config):
    torch.manual_seed(0)
    return DecoderOnlyLM.from_config(tiny_config).eval()


@pytest.mark.parametrize("language", ["hindi", "nepali"])
def test_full_size_parameter_count(language):
    """25,345,216 parameters, as reported in the Phase 2 report."""
    model = DecoderOnlyLM.from_config(LANGUAGES[language].load_model_config())
    assert model.num_parameters() == 25_345_216


def test_output_shape(tiny_model, tiny_config):
    x = torch.randint(0, tiny_config["vocab_size"], (3, 10))
    assert tiny_model(x).shape == (3, 10, tiny_config["vocab_size"])


def test_causal_masking_future_tokens_do_not_leak(tiny_model, tiny_config):
    """Changing token t must not change logits at positions < t."""
    x1 = torch.randint(0, tiny_config["vocab_size"], (1, 12))
    x2 = x1.clone()
    x2[0, 6] = (x2[0, 6] + 1) % tiny_config["vocab_size"]
    with torch.no_grad():
        l1, l2 = tiny_model(x1), tiny_model(x2)
    assert (l1[:, :6] - l2[:, :6]).abs().max().item() < 1e-6
    assert (l1[:, 6:] - l2[:, 6:]).abs().max().item() > 1e-6  # ...but it does change position >= t


def test_sequence_longer_than_context_is_rejected(tiny_model, tiny_config):
    with pytest.raises(ValueError):
        tiny_model(torch.zeros(1, tiny_config["seq_len"] + 1, dtype=torch.long))


def test_heads_must_divide_model_dim(tiny_config):
    with pytest.raises(ValueError):
        DecoderOnlyLM.from_config({**tiny_config, "num_heads": 5})


def test_loss_decreases_on_a_tiny_batch(tiny_config):
    torch.manual_seed(0)
    model = DecoderOnlyLM.from_config(tiny_config)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, tiny_config["vocab_size"], (4, 16))
    y = torch.roll(x, -1, dims=1)
    first = None
    for _ in range(40):
        loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, tiny_config["vocab_size"]), y.reshape(-1))
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5


def test_legacy_checkpoint_names_load(tiny_model, tiny_config):
    """Checkpoints with `final_norm` and without `causal_mask` (notebook-era) must load."""
    legacy = {}
    for k, v in tiny_model.state_dict().items():
        if k.endswith("causal_mask"):
            continue
        legacy[k.replace("ln_f.", "final_norm.")] = v
    fresh = DecoderOnlyLM.from_config(tiny_config)
    load_state_dict_compat(fresh, legacy, strict=True)
    x = torch.randint(0, tiny_config["vocab_size"], (1, 8))
    fresh.eval()
    assert torch.allclose(fresh(x), tiny_model(x), atol=1e-6)
