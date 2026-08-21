# Nepali Tokenizer Evaluation

This directory contains the final Phase 1 evaluation results for the
Nepali corpus and tokenizer.

## Final Corpus Statistics

| Metric | Value |
|---|---:|
| Documents | 10,056,762 |
| Words | 2,192,518,752 |
| Characters | 14,368,756,302 |
| Manual tokens | 100,000,000 |
| Non-manual tokens | 400,000,000 |
| Total tokens | 500,000,000 |
| Manual fraction | 20% |
| Vocabulary size | 31,744 |
| BPE merges | 31,488 |
| BPE training characters | 80,000,000 |

## Tokenizer

The final Nepali tokenizer uses a vocabulary of 31,744 tokens and
31,488 learned BPE merges.

The tokenizer was trained using an 80M-character training budget.

## Results

The machine-readable final statistics are available in:

`results.json`

The complete corpus is not stored in this repository. It is provided
separately through the project dataset storage.
