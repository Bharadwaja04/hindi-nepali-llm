[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

# Building and Analyzing Monolingual Transformer LMs in Two Indian Languages

Language Models and Agents — Monsoon 2026, Individual Project.

Two **fully independent** decoder-only Transformer language models
trained from scratch:

| Model | Language | Tier | Target tokens |
|-------|----------|------|---------------|
| Model H | Hindi | Higher-resource | ~500M |
| Model L | Nepali | Lower-resource | ~500M (or max feasible, documented) |

Separate data, tokenizer, vocabulary, and weights for each model. No
pretrained models, no pretrained tokenizers, no shared vocabulary.

## Google Drive (datasets, tokenizers, checkpoints)

All large artifacts live on Google Drive. Shareable link (permissions
set so TAs can open without requesting access):

**Main project folder:** https://drive.google.com/drive/folders/10EAbOVzHohdsI3gipC0qJCWVli7jE82N?usp=sharing

| Artifact | Drive path |
|----------|------------|
| Hindi raw (downloaded + manual) | `LMA_Project/hindi/raw/` |
| Hindi cleaned shards | `LMA_Project/hindi/processed/clean/` |
| Hindi splits | `LMA_Project/hindi/splits/` |
| Hindi tokenizer | `LMA_Project/hindi/tokenizer/hindi_tokenizer.json` (also in repo) |
| Hindi stats | `LMA_Project/hindi/stats/hindi_stats.json` (also in `reports/`) |
| Nepali raw (downloaded + manual OCR) | `LMA_Project/nepali/raw/` |
| Nepali cleaned shards | `LMA_Project/nepali/processed/clean/` |
| Nepali splits | `LMA_Project/nepali/splits/` |
| Nepali tokenizer | `LMA_Project/nepali/tokenizer/nepali_tokenizer.json` (also in repo) |
| Nepali stats | `LMA_Project/nepali/stats/nepali_stats.json` (also in `reports/`) |

> Phase 2/3 checkpoints (pretrained + finetuned) will be added here as
> the project progresses.

## Repository layout

```
README.md                        # this file — reproduction + Drive links
common/                          # shared (language-agnostic) pipeline code
    clean.py                     # streaming cleaner (NFC, HTML/URL strip, dedup, script filter)
    dedup.py                     # MinHash LSH near-duplicate removal
    splits.py                    # doc-level train/val/test split
    stats.py                     # corpus + tokenizer statistics
    collect.py                   # HuggingFace download (80% downloaded corpora)
    langfilter.py                # Unicode-block ratio language filter
hindi/
    tokenizer/                   # from-scratch byte-level BPE (Hindi)
        tokenizer.py  trainer.py  train.py  ...  hindi_tokenizer.json
nepali/
    tokenizer/                   # from-scratch byte-level BPE (Nepali)
        tokenizer.py  trainer.py  train.py  ...  nepali_tokenizer.json
reports/
    phase1_report.md             # Phase 1 report (stats, splits, tokenizer analysis)
    hindi_stats.json             # Hindi corpus statistics
    nepali_stats.json            # Nepali corpus statistics
scripts/
    phase1_pipeline.ipynb        # Colab notebook: full Phase 1 pipeline
```

## Reproduction steps (Phase 1)

Run on **Google Colab (CPU runtime is sufficient)**.

1. Open `scripts/phase1_pipeline.ipynb` in Colab.
2. Mount Drive + clone repo (cells 0–2).
3. **Stream A (Nepali):** clean the 22 GB `nepali_train.jsonl` +
   5 GB `nepali_test.jsonl` → cleaned shards. Clean Nepali manual OCR
   shards. (~2–4 h on Colab CPU.)
4. **Stream B (Hindi, parallel session):** download Hindi from
   HuggingFace (Wikipedia `hi`, mC4 `hi`) → clean. Clean Hindi manual
   web-crawl shards.
5. Merge downloaded + manual → near-dedup (MinHash) → splits (98/1/1).
6. Train Hindi BPE + Nepali BPE (32k vocab, 80M-char subsample, ~2–5 min
   each after the scalable trainer fix).
7. Encode full corpora → measure token counts + manual/downloaded split
   + unknown-token rate → write stats.
8. Commit + push to `phase-1` branch.

## Tokenizer

From-scratch **byte-level BPE** with Unicode-aware regex pre-tokenization
(NFC normalization). No pretrained merges, no tiktoken/GPT-4 vocabulary.

Architecture: `text → Unicode NFC → regex pre-tokenize → UTF-8 bytes →
BPE → token IDs`.

Train each language separately:

```bash
python -m hindi.tokenizer.train --language hindi \
    --data <hindi_train.jsonl> --vocab-size 32000 --char-budget 80000000 \
    --output hindi/tokenizer/hindi_tokenizer.json

python -m nepali.tokenizer.train --language nepali \
    --data <nepali_train.jsonl> --vocab-size 32000 --char-budget 80000000 \
    --output nepali/tokenizer/nepali_tokenizer.json
```

The trainer works on **unique pre-tokens weighted by frequency**
(minbpe/tiktoken approach) with incremental pair-count updates and a
max-heap with lazy deletion, so 32k merges on an 80M-char subsample
finish in minutes rather than weeks.
