# Nepali BPE Model

This directory contains the trained Nepali BPE tokenizer artifact used
for Phase 1 of the LMA project.

## Model

```text
nepali_bpe_31744.json
```

The model contains the Nepali-specific vocabulary and learned BPE merge
operations.

## Configuration

| Parameter | Value |
|---|---:|
| Language | Nepali |
| Vocabulary size | 31,744 |
| BPE merges | 31,488 |
| BPE training budget | 80,000,000 characters |

The tokenizer was trained independently on the Nepali corpus.

## Loading the Model

From the repository root:

```python
from nepali.tokenizer.tokenizer import Tokenizer

tokenizer = Tokenizer.load(
    "nepali/model/nepali_bpe_31744.json"
)
```

## Basic Usage

Encode text:

```python
text = "नेपाल एक सुन्दर देश हो।"

ids = tokenizer.encode(text)
print(ids)
```

Decode token IDs:

```python
decoded = tokenizer.decode(ids)
print(decoded)
```

Verify round-trip behavior:

```python
assert tokenizer.verify_round_trip(text)
```

## Training

The training implementation is located at:

`nepali/tokenizer/train.py`

The Phase 1 training configuration uses an 80M-character training
budget and a target vocabulary size of 31,744.

See:

`nepali/train/README.md`

for the training procedure and reproducibility instructions.

## Evaluation

See:

`nepali/eval/README.md`

for the final Phase 1 corpus and tokenizer evaluation statistics.

## Corpus

The complete Nepali corpus is not stored in this repository because of
its size. It is provided separately through the project dataset storage.

See the root-level `README.md` for corpus access and project-wide
reproduction instructions.
