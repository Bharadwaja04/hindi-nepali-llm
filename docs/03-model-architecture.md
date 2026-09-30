# Stage 3 — Model architecture (decoder-only Transformer, 25.3 M parameters)

**Code:** `src/indic_lm/model/transformer.py` · **Config:** `configs/{hindi,nepali}/model.json`
**Tests:** `tests/test_model.py` (parameter count, causal masking, legacy checkpoint loading)

## What this stage does

A GPT-style language model written in plain PyTorch (no `transformers` library, no pretrained weights). It maps a
sequence of token IDs to next-token logits.

```
ids ─► token emb + learned position emb ─► dropout
    ─► 6 × [ LN ─► causal multi-head attention ─► + residual ─► LN ─► GELU feed-forward ─► + residual ]
    ─► final LN ─► linear head (with bias) ─► logits over 31,744 tokens
```

| Setting | Value |
|---|---|
| Layers / heads / head dim | 6 / 8 / 36 (`d_model / heads = 288 / 8`) |
| `d_model`, `d_ff` | 288, 1440 (`d_ff = 5 · d_model`) |
| Context length | 128 tokens |
| Vocabulary | 31,744 (per language) |
| Dropout | 0.1 |
| Norm placement | pre-LayerNorm |
| Position information | learned absolute embeddings |
| Output head | untied from the input embedding, has a bias |
| Total parameters | **25,345,216** (identical for Hindi and Nepali; asserted in a unit test) |

## Equations

Let `X ∈ R^{T×d}` be the hidden states for `T` tokens.

**Attention** (per head `h`, head size `d_k = 36`):

```
Q, K, V = X W_Q, X W_K, X W_V
A_h     = softmax( Q_h K_hᵀ / √d_k  +  M )  V_h            M_ij = 0 if j ≤ i else −∞   (causal mask)
out     = concat(A_1 … A_8) W_O
```

The `1/√d_k` scaling keeps the dot products from growing with head size, which would push the softmax into a
saturated, near-one-hot regime with vanishing gradients. The mask `M` guarantees position `i` never sees `j > i`
(unit-tested by perturbing later tokens and checking that earlier logits do not change).

**Block (pre-LN):**

```
X ← X + Attn( LN(X) )
X ← X + W_2 · GELU( W_1 · LN(X) )
```

**GELU:** `GELU(x) = x · Φ(x)`, where `Φ` is the standard normal CDF.

**Training objective** (next-token cross-entropy over a window `x_1 … x_T`):

```
L = −(1/T) Σ_t log p_θ( x_{t+1} | x_{≤t} )        perplexity = exp(L)
```

## Parameter budget (computed from the code)

| Component | Parameters | Share |
|---|---:|---:|
| Token embedding (31,744 × 288) | 9,142,272 | 36.1 % |
| LM head (288 × 31,744 + bias) | 9,174,016 | 36.2 % |
| Attention, 6 layers | 1,997,568 | 7.9 % |
| Feed-forward, 6 layers | 4,987,008 | 19.7 % |
| Position embedding (128 × 288) | 36,864 | 0.1 % |
| LayerNorms | 7,488 | < 0.1 % |
| **Total** | **25,345,216** | 100 % |

**72.3 % of the parameters (18.3 M) sit in the input embedding and output head**, only 27.6 % in the
Transformer blocks. Practical consequences:

- The model is "small" in compute (blocks) but not in memory. Most gradient and optimizer state is spent on
  vocabulary rows, many of which are rarely seen.
- Tying the head to the embedding would save 9.1 M parameters (36 %) and usually costs little quality at this
  scale. The project kept them untied, which is a choice worth re-testing, not a proven optimum.
- Vocabulary size (stage 2) and model size are the same decision seen from two sides.

**Data-to-parameter ratio:** 500 M training tokens / 25.3 M parameters ≈ **19.7 tokens per parameter**, close to the
≈ 20:1 compute-optimal rule of thumb reported by the Chinchilla scaling study. That is a sensible balance for a
model this size, but because 72 % of parameters are lookup tables, the "effective" ratio for the blocks is much higher.

## Decisions, alternatives, and why

| Decision | Chosen | Alternatives | Why this one | Cost / what it gives up |
|---|---|---|---|---|
| Overall design | Decoder-only | Encoder-decoder (T5); encoder-only (BERT) | Next-token prediction on raw text needs no labels; the same network generates and can be scored by perplexity | No bidirectional context for classification tasks |
| Normalisation placement | Pre-LN | Post-LN (original Transformer) | Pre-LN trains stably without long warm-up tuning, important when a run cannot be repeated | Slightly lower final quality at equal depth is sometimes reported for pre-LN |
| Normalisation type | LayerNorm (with bias) | RMSNorm (LLaMA) | Standard, simple | RMSNorm is cheaper and just as good in most reports |
| Positions | Learned absolute | Sinusoidal; RoPE; ALiBi | Simplest; context is fixed at 128 | Cannot extrapolate beyond 128 tokens; RoPE/ALiBi generalise better and are now the default |
| Activation | GELU MLP | ReLU; SwiGLU (LLaMA/PaLM) | GELU is the GPT-2 default | SwiGLU is generally reported to give better quality per parameter |
| Attention | Standard multi-head | Multi-query / grouped-query; FlashAttention kernels | With T = 128 the O(T²) cost is negligible | GQA/MQA matter for serving with long contexts, not here. The attention here materialises the full T×T score matrix |
| Embedding/head | Untied | Tied | See above | +9.1 M parameters |
| Size | 6 × 288, 25.3 M | Larger | Fits training on a single free-tier GPU within the course budget | Limited capacity, so a limited ability to memorise facts or do multi-step reasoning |
| Dropout | 0.1 | 0 (typical for large single-epoch pretraining) | Cheap regularisation | With < 1 epoch over 500 M tokens, dropout probably costs more than it saves |

## Verification

`tests/test_model.py` checks, for both languages:

- the parameter count equals **25,345,216** exactly;
- the causal mask holds (changing token `j` never changes the output at positions `< j`);
- output shape is `(batch, T, vocab)`;
- a *legacy* checkpoint (old key names `final_norm.*`, no `causal_mask` buffer) loads through
  `load_state_dict_compat` with every tensor matched. The fine-tuned Hindi checkpoint from Phase 3 was loaded
  this way with all keys matching, so the refactor did not break the released weights.

## Limitations

- The design is the 2019 GPT-2 recipe. Modern small models would use RoPE, RMSNorm, SwiGLU and tied embeddings.
  These are deliberate simplifications, listed as upgrade candidates and not tested here.
- A 128-token context (about 60 words with this tokenizer, see stage 2) limits every downstream task. Long
  reasoning chains do not fit.
- No ablations were run. None of the alternatives above was trained and compared, so the "why" column reflects
  reasoning and convention, not measured evidence, unlike the tokenizer vocabulary-size study.
