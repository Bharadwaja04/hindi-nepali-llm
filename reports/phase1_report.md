# Phase 1 Report — Data Collection and Tokenizer Construction

**Languages:** Hindi (Model H, higher-resource) · Nepali (Model L, lower-resource)
**Deadline:** 19 August 2026

## 1. Language Selection

**Model H: Hindi** — selected as the higher-resource non-English Indian
language due to abundant public text across news, literature, Wikipedia,
educational and general-domain sources.

**Model L: Nepali** — one of the explicitly permitted lower-resource
languages. The project investigates how corpus scale, quality, and
tokenizer characteristics affect a monolingual Nepali LM.

The two models are **fully independent**: separate corpora, tokenizers,
vocabularies, and (in Phase 2) weights. No data, merges, or checkpoints
are shared.

## 2. Dataset Construction

Each model has its own monolingual corpus. Target: ~500M training tokens
per language after tokenization. Per the project spec, ≥20% of final
training tokens must come from **manual collection** (OCR / scraping /
transcription); the remainder may be downloaded from public corpora.

### 2.1 Sources

#### Hindi (Model H)
- **Downloaded (~80%):**
  - Wikimedia Wikipedia (`20231101.hi`) — general encyclopedic text
  - mC4 (`allenai/c4`, multilingual, `hi`) — web-crawled general text
  - (OSCAR `hi` as an optional additional source)
- **Manual (~20%):**
  - Self-scraped Hindi web pages (news, wikisource, articles) →
    `hindi/raw/manual/` → OCR'd/scraped → `hindi_webcrawl_raw_batch_*.jsonl`

#### Nepali (Model L)
- **Downloaded (~80%):**
  - `nepali_train.jsonl` (22.3 GB) + `nepali_test.jsonl` (5.1 GB) —
    large web-crawled Nepali corpora
- **Manual (~20%):**
  - OCR of books from **Nepal National Library** (20 OCR shards)
  - Text from **Sahityaras** (literature collection)
  - Text from **CDC** (curriculum development centre)
  - Scraped **Kantipur** news articles
  - Scraped general Nepali web pages

### 2.2 Preprocessing pipeline

All cleaning is done by `common/clean.py` (shared, language-agnostic;
language selected via `--lang`). Steps per document:

1. **Unicode NFC normalization** (`unicodedata.normalize("NFC", ...)`)
2. **Strip non-printable control characters**
3. **Strip HTML tags, markdown links, URLs, email addresses**
4. **De-spam:** collapse 5+ repeated chars, collapse 5+ repeated words
5. **Whitespace normalization** (single spaces, collapse 3+ blank lines)
6. **Length filter:** drop docs < 80 chars (downloaded) / < 40 chars
   (manual OCR, which has shorter segments)
7. **Script filter:** keep docs where ≥55% of non-space characters are
   Devanagari (Unicode block U+0900–U+097F). This removes
   English-dominant pages and mis-OCR'd garbage.
8. **Repetition filter:** drop docs where unique-word ratio < 0.65
   (catches boilerplate replays and gibberish)
9. **Exact dedup:** SHA-1 of normalized text; bloom filter (optional)
   for RAM-bounded dedup on the 22 GB file
10. **Near-dedup (post-clean):** MinHash LSH (64 permutations, 8 bands,
    threshold 0.9, 5-word shingles) via `common/dedup.py`
11. **Splits:** doc-level train/val/test = 98/1/1, seeded shuffle
    (`common/splits.py`, seed=17), no leakage

### 2.3 Corpus statistics

> Numbers below are filled in by `common/stats.py` after the pipeline
> runs. See `reports/hindi_stats.json` and `reports/nepali_stats.json`
> for the authoritative values.

| Metric | Hindi (H) | Nepali (L) |
|--------|-----------|------------|
| Documents (train) | TBD | TBD |
| Total characters | TBD | TBD |
| Total UTF-8 bytes | TBD | TBD |
| Total tokens (BPE) | TBD | TBD |
| Tokens (millions) | TBD | TBD |
| Manual tokens (%) | TBD | TBD |
| Downloaded tokens (%) | TBD | TBD |
| Avg chars/token | TBD | TBD |
| Avg bytes/token | TBD | TBD |
| Vocab size | 32000 | 32000 |
| Unknown-token rate (held-out) | TBD | TBD |

**Manual vs downloaded token split** is computed by `common/stats.py`
using the `source` field tagged during cleaning (`manual` or
`downloaded`). The ≥20% manual requirement is checked per language; if a
language falls short, the exact fraction and a justification are
documented here.

**Nepali shortfall justification (if applicable):** Nepali is a
lower-resource language with a smaller public text footprint. If the
final token count is below 500M, we report the exact count and note that
all feasible public + manual sources were collected, per the project's
allowance for lower-resource languages.

## 3. Tokenizer Training

Separate from-scratch **byte-level BPE** tokenizers per language. No
pretrained merges, no tiktoken/GPT-4 vocabulary.

### 3.1 Architecture

```
text → Unicode NFC → regex pre-tokenize → UTF-8 bytes → BPE → token IDs
```

- **Pre-tokenization:** Unicode-aware regex
  (`\s+ | \p{L}[\p{M}\p{L}]* | \p{N}+ | [^\p{L}\p{N}\s]`) keeps
  Devanagari conjuncts and combining marks together.
- **Base vocabulary:** 256 byte IDs.
- **Merges:** `vocab_size - 256` byte-pair merges learned from the
  training corpus.
- **Trainer optimization:** works on **unique pre-tokens weighted by
  frequency** with incremental pair-count updates and a max-heap with
  lazy deletion. Trains on an 80M-character representative subsample
  (standard practice — BPE merges are statistically representative at
  this scale; the full 500M corpus is only *encoded*, not *trained on*).

### 3.2 Vocabulary size selection

Vocabulary size = 32,000 for both languages (in the recommended "tens
of thousands" range). The choice is justified by **fertility** (avg
chars/token) and **unknown-token rate** on held-out text, reported in
the stats table above.

### 3.3 Tokenization examples

> Filled in by the notebook's examples cell. See §3 of the notebook
> output.

### 3.4 Token-frequency statistics

> Filled in by `common/stats.py` / `tokenizer.tokenization_stats()`.

## 4. Train / Validation / Test Splits

Doc-level splits (98/1/1), seeded deterministic shuffle, no leakage.
Files:
- `hindi/splits/hindi_train.jsonl`, `hindi_val.jsonl`, `hindi_test.jsonl`
- `nepali/splits/nepali_train.jsonl`, `nepali_val.jsonl`, `nepali_test.jsonl`
- Held-out samples: `*_heldout.jsonl` (500 docs each, for tokenizer
  unknown-rate evaluation)

## 5. Deliverables

- [x] Dataset collection scripts (`common/collect.py`, manual scrapers)
- [x] Dataset preprocessing pipelines (`common/clean.py`, `common/dedup.py`)
- [x] Per-language dataset statistics (`reports/{hindi,nepali}_stats.json`)
- [x] Per-language train/val/test splits (`{hindi,nepali}/splits/`)
- [x] Tokenizer training code (`{hindi,nepali}/tokenizer/train.py`)
- [x] Vocabulary files (inside tokenizer JSON)
- [x] Tokenizer model files (`{hindi,nepali}/tokenizer/{lang}_tokenizer.json`)
