"""Decoder-only causal Transformer language model (GPT-style, pre-LayerNorm).

One implementation serves both languages. The original project kept two
copies (``HindiLanguageModel`` / ``NepaliLanguageModel``) that differed only in
class name and formatting; parameter names below are identical to those, so
checkpoints trained with either copy load unchanged.

Notation (see docs/03-model-architecture.md for the equations):
    B batch, T sequence length, C = d_model, H heads, d_h = C / H.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product attention with an upper-triangular mask."""

    def __init__(self, d_model: int, num_heads: int, seq_len: int, dropout: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # True above the diagonal => position i may not attend to j > i.
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))

        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        y = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_dropout(self.out_proj(y))


class FeedForward(nn.Module):
    """Position-wise MLP: Linear -> GELU -> Dropout -> Linear -> Dropout."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    """Pre-LN block: x + Attn(LN(x)), then x + FFN(LN(x))."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, seq_len: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class DecoderOnlyLM(nn.Module):
    """Learned token + learned absolute position embeddings, untied LM head."""

    def __init__(
        self,
        vocab_size: int = 31744,
        d_model: int = 288,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1440,
        seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff, seq_len, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=True)

    @classmethod
    def from_config(cls, config: dict) -> "DecoderOnlyLM":
        keys = ("vocab_size", "d_model", "num_heads", "num_layers", "d_ff", "seq_len", "dropout")
        return cls(**{k: config[k] for k in keys if k in config})

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, T = token_ids.shape
        if T > self.seq_len:
            raise ValueError(f"Sequence length {T} exceeds maximum length {self.seq_len}")

        positions = torch.arange(T, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))


# Backwards-compatible names used by the original per-language code.
HindiLanguageModel = DecoderOnlyLM
NepaliLanguageModel = DecoderOnlyLM


def load_state_dict_compat(model: nn.Module, state_dict: dict, strict: bool = True):
    """Load checkpoints produced by the original notebooks and scripts.

    Handles two historical differences: some checkpoints name the final
    LayerNorm ``final_norm`` instead of ``ln_f``, and the attention mask
    buffer (``causal_mask``) is present in some and absent in others; it is
    a constant so it is safe to ignore either way.
    """
    fixed = {}
    for key, value in state_dict.items():
        if key.startswith("final_norm."):
            key = "ln_f." + key[len("final_norm."):]
        fixed[key] = value

    model_keys = set(model.state_dict().keys())
    mask_keys = {k for k in model_keys if k.endswith("causal_mask")}
    for key in mask_keys - set(fixed):
        fixed[key] = model.state_dict()[key]
    fixed = {k: v for k, v in fixed.items() if k in model_keys or strict}
    return model.load_state_dict(fixed, strict=strict)
