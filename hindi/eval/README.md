# Hindi Tokenizer Evaluation

This directory contains the final Phase 1 evaluation results for the
Hindi corpus and tokenizer.

## Final Corpus Statistics

| Metric | Value |
|---|---:|
| Documents | 2,312,791 |
| Words | 886,594,715 |
| Characters | 5,393,815,951 |
| Manual tokens | 100,000,000 |
| Non-manual tokens | 400,000,000 |
| Total tokens | 500,000,000 |
| Manual fraction | 20% |
| Vocabulary size | 31,744 |
| BPE merges | 31,488 |
| BPE training characters | 80,000,000 |

## Tokenizer

The final Hindi tokenizer uses a vocabulary of 31,744 tokens and
31,488 learned BPE merges.

The tokenizer was trained using an 80M-character training budget.

## Results

The machine-readable final statistics are available in:

`results.json`

The complete corpus is not stored in this repository. It is provided
separately through the project dataset storage.
