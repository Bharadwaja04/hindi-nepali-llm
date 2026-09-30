# Stage 4 — Pretraining

**Code:** `src/indic_lm/training/pretrain.py`, `training/utils.py`, `data/streams.py`
**Configs:** `configs/{hindi,nepali}/model.json` · **Hindi training log:** `reports/phase2/hindi/training_log.jsonl`
**Tests:** `tests/test_smoke_training.py` (logging, checkpointing, resume equivalence), `tests/test_pipeline_units.py` (LR schedule, batching)

## What this stage does

Trains the 25.3 M-parameter model (stage 3) from random initialisation on ≈ 500 M tokens per language, with plain
next-token prediction.

```
token stream (20 % manual + 80 % downloaded, exact budgets) ─► windows of 128 ─► micro-batches of 16
 ─► 4 micro-batches accumulated = 1 optimiser step of 8,192 tokens ─► AdamW + warmup/cosine LR ─► checkpoint
```

| Setting | Value |
|---|---|
| Tokens per optimiser step | 16 (micro-batch) × 4 (accumulation) × 128 (context) = **8,192** |
| Steps | 500,000,000 / 8,192 → **61,035** (499,998,720 tokens, under one epoch) |
| Optimiser | AdamW, β = (0.9, 0.95), weight decay 0.1 |
| Learning rate | peak 3e-4, 2,000 warmup steps, then cosine decay (see the discrepancy below) |
| Precision | FP16 autocast with `GradScaler` (GPU only) |
| Gradient clipping | global norm 1.0 |
| Checkpoints | every 1,000 steps (model, optimiser, scheduler, scaler, config) |
| Hindi wall-clock | ≈ 4.98 h at ≈ 28–29 k tokens/s (from the log) |

## Equations

**Objective:** mean next-token cross-entropy, `L = −(1/N) Σ log p_θ(x_{t+1} | x_{≤t})`. Each optimiser step averages the
loss over 4 micro-batches (`loss / accum`), so the logged number is a true mean and gradient accumulation matches
a batch of 64 sequences.

**AdamW** (decoupled weight decay), with gradient `g_t`, learning rate `η_t`, decay `λ`:

```
m_t = β1 m_{t−1} + (1−β1) g_t          v_t = β2 v_{t−1} + (1−β2) g_t²
θ_t = θ_{t−1} − η_t · ( m̂_t / (√v̂_t + ε)  +  λ θ_{t−1} )
```

`β2 = 0.95` (instead of the default 0.999) makes the second-moment estimate adapt faster. It is the value GPT-3 and
LLaMA use, and it helps stability when the loss surface changes quickly early in training.

**Learning-rate schedule:**

```
warmup   (t ≤ 2000):   η_t = η_max · t / 2000
cosine   (t > 2000):   η_t = η_min + (η_max − η_min) · ½ · (1 + cos(π · (t−2000) / (T−2000)))     T = 61,035
```

**Gradient clipping:** if `‖g‖₂ > 1`, rescale `g ← g / ‖g‖₂`.

**Sanity anchors.** A model that guesses uniformly has loss `ln(31,744) = 10.37` (perplexity 31,744). The first logged
Hindi step (100, still in warmup) is 7.42, and the last is 2.50 (perplexity 12.2).

## Results (Hindi log, mean of logged training loss per range)

| Steps | Mean train loss |
|---|---:|
| 100 – 1,000 | 5.34 |
| 1,001 – 6,000 | 4.05 |
| 6,001 – 20,000 | 2.98 |
| 20,001 – 40,000 | 2.63 |
| 40,001 – 61,000 | 2.56 |

Final held-out results (`reports/phase2`):

| | Hindi | Nepali |
|---|---:|---:|
| Validation loss | 2.5498 | 2.6047 |
| Perplexity (`exp(loss)`) | 12.80 | 13.53 |
| Bits/token (`loss / ln 2`) | 3.679 | 3.758 |
| Evaluated on | 5,322,443 tokens | 359,396 tokens (1,000 docs, seed 42) |

Both perplexity and bits/token were recomputed from the reported loss in a unit test and agree (`exp(2.549785) = 12.8044`).

**Reading the numbers.**
- Training loss (≈ 2.50 at the last logged step) and validation loss (2.55) are close, so there is no sign of
  overfitting, which is expected for < 1 epoch with 25 M parameters.
- The curve is still falling slowly at the end (2.63 → 2.56 over the last 40 k steps), so the model is
  capacity- and token-limited, not converged in a strong sense. It is not evidence that more data would not help.
- **Perplexity cannot be compared across the two languages or with other papers.** The two models use different
  tokenizers and different test sets, and perplexity is per *token*, not per character or word. Hindi 12.8 vs Nepali
  13.5 is not a claim that Hindi is "better modelled".
- Perplexity ≈ 13 means that on average the model is as uncertain as a uniform choice among ~13 tokens, and since
  about 45 % of tokens are trivially predictable spaces (stage 2), the real uncertainty on content tokens is higher than
  the headline suggests.

## Decisions, alternatives, and why

| Decision | Chosen | Alternatives | Why this one | Cost |
|---|---|---|---|---|
| Optimiser | AdamW | SGD+momentum; Adafactor; Lion; Muon | Standard for Transformer LMs; the decoupled decay is correct for adaptive methods | Two extra states per parameter (2× memory) |
| Learning rate | 3e-4 peak | Sweep 1e-4 … 1e-3 | Typical for this scale | **No LR sweep was run**; the value is conventional, not tuned |
| Warmup | 2,000 steps linear | none; longer | Adam's second-moment estimate is noisy in the first steps | 3 % of the run |
| Decay | cosine | linear; constant; WSD (warmup-stable-decay); inverse-sqrt | Smooth, no extra hyper-parameter besides the floor | Needs the total step count up front |
| Batch | 8,192 tokens | larger | Fits a free-tier GPU via accumulation | Small by modern standards, so gradient noise is high |
| Precision | FP16 + GradScaler | BF16 (no scaler, wider range); FP32 | The available GPUs (T4-class) lack BF16 | FP16 needs loss scaling; overflow steps are skipped silently |
| Clipping | 1.0 | none; adaptive | Guards against loss spikes | — |
| Epochs | < 1 | multi-epoch | Fresh tokens every step, so no memorisation | Data pipeline must stream ≥ 500 M tokens |
| Data mix | 20 % manual / 80 % downloaded, exact budgets | uniform sampling; upweight high-quality | Course requirement | Fixed mix, not tuned |
| Resume | Skip consumed micro-batches, restore optimiser + scheduler + scaler | restart data from the beginning | Same data order after a Colab disconnect. Verified by a test: 3 steps + resume + 3 steps equals 6 straight steps | Skipping replays the tokenizer through the consumed prefix (slow for late steps) |

## Known problems (found while consolidating; also in [known-issues.md](known-issues.md))

1. **The released Hindi run and the repo's scheduler code do not match.** The code (`create_scheduler`, `eta_min = 0.1 · η_max`)
   ends at 3e-5; simulating the exact code for all 61,035 steps gives 3.0e-5 at the end. The shipped Hindi training log
   shows the LR falling to **2.6e-10** at step 61,000, which is exactly a pure cosine to zero. The log was therefore
   produced by a different schedule from the current code (the reason is unknown; probably an earlier version).
   The results above are real, but re-running this repo will not reproduce the exact LR curve. Decide which is
   intended and fix the other.
2. **Weight decay is applied to everything**, including biases, LayerNorm parameters and embeddings
   (`optimizer(model.parameters())`). The usual practice is to decay only weight matrices. The effect is probably small, but
   this is not the standard recipe and was not ablated.
3. **"Best checkpoint at step 6,000"** in the phase report cannot be right: the mean training loss over steps
   1,001–6,000 is 4.05 against 2.56 at the end. The claim most likely refers to a different selection rule; the final
   checkpoint is the one to use, and the doc set treats it that way.
4. **No validation loss during the original training run.** The log has training loss only. `pretrain.py` now supports
   `--eval-every`, but the released curves cannot show a train/validation gap over time.
5. **Bits-per-byte is inconsistent.** The report lists BPB 0.357 alongside 3.679 bits/token, which implies
   3.679 / 0.357 = 10.3 bytes per token. A Devanagari character is 3 UTF-8 bytes and this tokenizer produces ≈ 2.3 characters per token, so
   the real figure cannot exceed about 7 bytes per token. **Do not quote the BPB value.** Bits/token and perplexity are
   consistent and are the numbers to use.

## Reproducing

```bash
python -m indic_lm.training.pretrain --language hindi \
    --manual-data <manual .jsonl ...> --nonmanual-data <downloaded .jsonl ...> \
    --val-data <validation .jsonl ...> --eval-every 1000 \
    --checkpoint-dir models/hindi --log logs/hindi.jsonl
# resume after an interruption:      add  --resume models/hindi/checkpoint_step_030000.pt
# smoke test (a few CPU steps):      add  --max-steps 5
```

(`--help` lists every flag; `--config` and `--tokenizer` default to the per-language files under `configs/` and
`artifacts/`. The full run needs the corpus described in `data/README.md` and a GPU; ≈ 5 h on the hardware used for
the original run.)
