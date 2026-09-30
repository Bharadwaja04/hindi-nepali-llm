# artifacts/

Small, versioned outputs.

| Path | Contents |
|---|---|
| `tokenizers/{hindi,nepali}_bpe_31744.json` | Trained tokenizers (256 byte tokens + 31,488 merges). Load with `Tokenizer.load` |
| `tokenizers/tokenizer_report.json` | Output of `make tokenizer-check` (fertility, whitespace share, cross-language cost, round-trip results) |
| `eval/hindi_reasoning_qa.json` | Held-out Hindi reasoning evaluation, answer-only prompt, 957 questions |
| `eval/hindi_reasoning_cot_first300.json` | Same, reasoning-first prompt, first 300 questions |
