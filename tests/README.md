# tests/

CPU-only, no downloads, seconds to run: `make test`.

| File | Covers |
|---|---|
| `test_tokenizer.py` | Round-trips (including Latin, digits, emoji), NFC, released-artifact structure (31,744 = 256 + 31,488) |
| `test_model.py` | Exact parameter count (25,345,216) for both languages, causal-mask leak test, shape, legacy checkpoint loading |
| `test_pipeline_units.py` | Perplexity / bits-per-token closed forms, Distinct-n, repetition, batching alignment, LR warmup + cosine, checkpoint round-trip, reasoning-evaluation helpers and baselines |
| `test_smoke_training.py` | Tiny end-to-end pretraining, resume equivalence, fine-tuning padding and updates |
