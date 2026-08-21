# Hindi Dataset Pipeline

This directory contains the Phase 1 Hindi corpus collection,
preprocessing, deduplication, language filtering, statistics,
and split definitions.

## Pipeline

```text
Raw sources
    ↓
Collection manifest
    ↓
Validation
    ↓
Unicode NFC normalization
    ↓
Whitespace normalization
    ↓
Language/script filtering
    ↓
Exact document deduplication
    ↓
Corpus statistics
    ↓
Train / validation / test split
    ↓
Hindi BPE tokenizer

## Data provenance

Each document retains its source category:
- manual
- non_manual

Large datasets are stored on Google Drive and are not committed
to Git.

## Reproducibility
All preprocessing operations use deterministic processing.
Splits use a fixed random seed.

## Important
The preprocessing pipeline must be run before tokenizer training
when reproducing the corpus from raw sources.


Make the Nepali README identical except for Hindi → Nepali.

---

# 11. Root README needs to be much stronger

Your current README should explicitly state the independence requirement.

Use this section:

```markdown
# LMA — Building and Analyzing Monolingual Transformer LMs

This project implements two completely independent monolingual
language-model pipelines:

- Model H — Hindi
- Model L — Nepali

The models do not share:

- datasets
- documents
- tokenizers
- vocabularies
- model weights

## Phase 1

Phase 1 covers:

1. Dataset collection
2. Dataset preprocessing
3. Unicode normalization
4. Invalid-document removal
5. Duplicate removal
6. Language filtering
7. Corpus statistics
8. Train/validation/test splits
9. Independent BPE tokenizer training
10. Tokenizer evaluation

## Repository

```text
hindi/
    data/
    tokenizer/
    model/
    train/
    eval/
    configs/

nepali/
    data/
    tokenizer/
    model/
    train/
    eval/
    configs/

report/