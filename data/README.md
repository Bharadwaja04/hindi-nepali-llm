# data/

Only the small held-out reasoning **test** sets are tracked (`reasoning/{hindi,nepali}/splits/test.jsonl`).

Everything else is too large or regenerable and is git-ignored:

| What | Size / source | How to obtain |
|---|---|---|
| Raw and cleaned corpora (≈ 500 M training tokens per language) | tens of GB; public web corpora (mC4, OSCAR, Wikipedia, IndicCorp) plus manually collected material | Rebuild with `indic_lm.data.collect`, then clean, filter, deduplicate and split ([docs/01](../docs/01-data-pipeline.md)); the original copies are on the project's Google Drive |
| Reasoning train / validation splits | small, synthetic | `make reasoning-data` (seed 42) |

Reasoning-record fields: `id`, `language`, `category`, `question`, `reasoning`, `answer`, `text`.
