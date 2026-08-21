[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

# Building and Analyzing Monolingual Transformer LMs in Two Indian Languages

Language Models and Agents — Monsoon 2026, Individual Project.

This project builds two fully independent language-model pipelines for
Hindi and Nepali. Each language has its own corpus, preprocessing
pipeline, tokenizer, vocabulary, and model artifacts.

No pretrained language model, pretrained tokenizer, or shared vocabulary
is used.

| Model | Language | Resource tier | Target corpus |
|---|---|---|---:|
| Model H | Hindi | Higher-resource | 500M tokens |
| Model L | Nepali | Lower-resource | 500M tokens |

---

## Phase 1 Status

Phase 1 focuses on corpus construction, preprocessing, tokenizer
training, full-corpus tokenization, and tokenizer evaluation.

### Final corpus targets

| Metric | Hindi | Nepali |
|---|---:|---:|
| Documents | 2,312,791 | 10,056,762 |
| Words | 886,594,715 | 2,192,518,752 |
| Characters | 5,393,815,951 | 14,368,756,302 |
| Manual tokens | 100,000,000 | 100,000,000 |
| Non-manual tokens | 400,000,000 | 400,000,000 |
| Total tokens | 500,000,000 | 500,000,000 |
| Manual fraction | 20% | 20% |
| Vocabulary size | 31,744 | 31,744 |
| BPE merges | 31,488 | 31,488 |
| BPE training characters | 80,000,000 | 80,000,000 |

The large corpus files are not stored in GitHub. They are provided
through the project Google Drive.

---

## Google Drive

All large corpus artifacts and generated datasets are stored separately
from this repository.

**Main project folder:**

https://drive.google.com/drive/folders/10EAbOVzHohdsI3gipC0qJCWVli7jE82N?usp=sharing

### Dataset organization

```text
LMA_Project/
├── hindi/
│   ├── raw/
│   ├── processed/
│   │   ├── clean/
│   │   └── tokenized/
│   └── tokenizer/
│
└── nepali/
    ├── raw/
    ├── processed/
    │   ├── clean/
    │   └── tokenized/
    └── tokenizer/
```

The corpus is constructed from downloaded language resources together
with manually collected language-specific material.

The large raw, cleaned, and tokenized corpus files are intentionally
excluded from GitHub because of their size.

---

## Repository Structure

```text
.
├── README.md
│
├── common/
│   ├── __init__.py
│   ├── clean.py
│   ├── collect.py
│   ├── dedup.py
│   ├── langfilter.py
│   ├── splits.py
│   └── stats.py
│
├── hindi/
│   ├── data/
│   │   ├── collect.py
│   │   ├── deduplicate.py
│   │   ├── language_filter.py
│   │   ├── preprocess.py
│   │   ├── splits.py
│   │   ├── statisitics.py
│   │   └── README.md
│   │
│   ├── tokenizer/
│   │   ├── tokenizer.py
│   │   ├── trainer.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── bpe.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   ├── pretokenizer.py
│   │   ├── vocab.py
│   │   ├── config.py
│   │   ├── test_tokenizer.py
│   │   └── README.md
│   │
│   ├── train/
│   │   └── README.md
│   │
│   ├── model/
│   │   ├── README.md
│   │   └── hindi_bpe_31744.json
│   │
│   └── eval/
│       ├── README.md
│       └── results.json
│
├── nepali/
│   ├── data/
│   │   ├── collect.py
│   │   ├── clean.py
│   │   ├── dedup.py
│   │   ├── langfilter.py
│   │   ├── splits.py
│   │   ├── stats.py
│   │   └── README.md
│   │
│   ├── tokenizer/
│   │   ├── tokenizer.py
│   │   ├── trainer.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── bpe.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   ├── pretokenizer.py
│   │   ├── vocab.py
│   │   ├── config.py
│   │   ├── test_tokenizer.py
│   │   └── README.md
│   │
│   ├── train/
│   │   └── README.md
│   │
│   ├── model/
│   │   ├── README.md
│   │   └── nepali_bpe_31744.json
│   │
│   └── eval/
│       ├── README.md
│       └── results.json
│
└── reports/
    └── phase1_report.md
```

---

## Data Pipeline

The corpus pipeline is language-specific while following the same
general processing stages.

```text
Raw sources
    |
    v
Collection
    |
    v
Cleaning
    |
    v
Unicode normalization
    |
    v
Language filtering
    |
    v
Deduplication
    |
    v
Train / validation / test split
    |
    v
Corpus statistics
    |
    v
BPE tokenizer training
    |
    v
Full-corpus tokenization
    |
    v
Final Phase 1 corpus
```

The processing code is available under the corresponding `data/`
directories and the shared `common/` directory.

---

## Corpus Construction

The final Phase 1 target is 500M tokens for each language.

The corpus is divided into:

- 100M manually collected/curated tokens
- 400M non-manual tokens

This gives a final manual fraction of approximately 20%.

The full corpus is maintained in Google Drive rather than committed to
GitHub.

### Hindi

The Hindi corpus contains approximately:

- 2.31M documents
- 886.59M words
- 5.39B characters
- 500M final tokens

### Nepali

The Nepali corpus contains approximately:

- 10.06M documents
- 2.19B words
- 14.37B characters
- 500M final tokens

---

## Tokenizer

Both languages use an independently trained, from-scratch,
Unicode-aware byte-level BPE tokenizer.

Architecture:

```text
Text
  |
  v
Unicode NFC normalization
  |
  v
Regex pre-tokenization
  |
  v
UTF-8 bytes
  |
  v
BPE merge operations
  |
  v
Token IDs
```

The Hindi and Nepali tokenizers have separate:

- training corpora
- vocabularies
- merge tables
- tokenizer artifacts

No pretrained tokenizer vocabulary or GPT/tiktoken vocabulary is used.

---

## BPE Training Configuration

The Phase 1 tokenizer configuration is:

| Parameter | Hindi | Nepali |
|---|---:|---:|
| Vocabulary size | 31,744 | 31,744 |
| BPE merges | 31,488 | 31,488 |
| Training character budget | 80M | 80M |
| Base byte vocabulary | 256 | 256 |

The BPE merge-training budget is 80M characters.

This does not mean that the final corpus contains only 80M characters.
The 80M-character sample is used to learn the tokenizer vocabulary
and merge operations. The trained tokenizer is then applied to the
full Phase 1 corpus.

---

## Training

Install the tokenizer dependency:

```bash
pip install regex
```

Train Hindi:

```bash
python -m hindi.tokenizer.train     --language hindi     --data <PATH_TO_HINDI_TRAINING_DATA>     --vocab-size 31744     --output hindi/model/hindi_bpe_31744.json     --char-budget 80000000
```

Train Nepali:

```bash
python -m nepali.tokenizer.train     --language nepali     --data <PATH_TO_NEPALI_TRAINING_DATA>     --vocab-size 31744     --output nepali/model/nepali_bpe_31744.json     --char-budget 80000000
```

The two tokenizers must be trained independently.

---

## Loading a Trained Tokenizer

Hindi:

```python
from hindi.tokenizer.tokenizer import Tokenizer

tokenizer = Tokenizer.load(
    "hindi/model/hindi_bpe_31744.json"
)
```

Nepali:

```python
from nepali.tokenizer.tokenizer import Tokenizer

tokenizer = Tokenizer.load(
    "nepali/model/nepali_bpe_31744.json"
)
```

Example encoding:

```python
text = "भारत एक सुंदर देश है।"
ids = tokenizer.encode(text)
print(ids)
```

Example decoding:

```python
decoded = tokenizer.decode(ids)
print(decoded)
```

Round-trip verification:

```python
assert tokenizer.verify_round_trip(text)
```

---

## Evaluation

Final corpus-level evaluation results are stored at:

```text
hindi/eval/results.json
nepali/eval/results.json
```

The corresponding documentation is available in:

```text
hindi/eval/README.md
nepali/eval/README.md
```

The evaluation includes corpus size, token counts, vocabulary size,
merge count, and training character budget.

---

## Vocabulary Size and Fertility Analysis

A vocabulary-size experiment is being evaluated using:

```text
8,192
16,384
24,576
32,768
49,152
65,536
```

The experiment evaluates Hindi and Nepali using the same evaluation
procedure and a 10,000-document evaluation sample per language.

The primary comparison is:

```text
Vocabulary size  <->  Fertility
```

where:

```text
Fertility = number of tokens / number of words
```

The final measured fertility values and figures will be added to the
Phase 1 report after the experiment completes.

---

## Phase 1 Reports

The main Phase 1 report is:

`reports/phase1_report.md`

It documents the corpus construction, preprocessing, tokenizer design,
training configuration, statistics, and evaluation.

Additional vocabulary-size and fertility figures will be stored under
`reports/` once the experiment is finalized.

---

## Reproducibility

A complete Phase 1 reproduction can be performed using the source code
in this repository together with the large corpus stored in Google
Drive.

Recommended environment:

```text
Python 3.x
Google Colab or Linux environment
regex
```

The large corpus should be mounted from Google Drive and should not be
copied into the Git repository.

The repository contains the tokenizer implementation and trained
tokenizer artifacts required to inspect and reproduce tokenizer
behavior.

---

## Project Scope

Phase 1 covers:

- corpus collection
- corpus preprocessing
- cleaning
- language filtering
- deduplication
- train/validation/test splitting
- corpus statistics
- tokenizer training
- full-corpus tokenization
- vocabulary-size analysis
- tokenizer evaluation

Later phases will cover the decoder-only Transformer language models,
pretraining, evaluation, and fine-tuning.

---

## Current Phase 1 Deliverables

```text
Hindi
  ✓ Corpus pipeline
  ✓ Cleaning and filtering
  ✓ Deduplication
  ✓ BPE tokenizer implementation
  ✓ 31,744-token vocabulary
  ✓ 31,488 BPE merges
  ✓ Trained tokenizer artifact
  ✓ Evaluation results

Nepali
  ✓ Corpus pipeline
  ✓ Cleaning and filtering
  ✓ Deduplication
  ✓ BPE tokenizer implementation
  ✓ 31,744-token vocabulary
  ✓ 31,488 BPE merges
  ✓ Trained tokenizer artifact
  ✓ Evaluation results

Analysis
  ✓ Phase 1 corpus statistics
  ✓ Tokenizer statistics
  ⏳ Vocabulary-size × fertility experiment
  ⏳ Final fertility figures and plots
```

---

## Important Notes

The corpus itself is intentionally not included in this GitHub
repository because of its size.

The Google Drive folder linked above is the source for the large
dataset artifacts.

The repository is intended to contain the reproducible code,
configuration, tokenizer artifacts, evaluation outputs, and project
documentation.
