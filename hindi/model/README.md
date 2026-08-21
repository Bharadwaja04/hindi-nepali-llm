# Hindi BPE Model

This directory contains the trained Hindi BPE tokenizer artifact used
for Phase 1 of the LMA project.

## Model

```text
hindi_bpe_31744.json
```

The model contains the Hindi-specific vocabulary and learned BPE merge
operations.

## Configuration

| Parameter | Value |
|---|---:|
| Language | Hindi |
| Vocabulary size | 31,744 |
| BPE merges | 31,488 |
| BPE training budget | 80,000,000 characters |

The tokenizer was trained independently on the Hindi corpus.

## Loading the Model

From the repository root:

```python
from hindi.tokenizer.tokenizer import Tokenizer

tokenizer = Tokenizer.load(
    "hindi/model/hindi_bpe_31744.json"
)
```

## Basic Usage

Encode text:

```python
text = "भारत एक सुंदर देश है।"

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

`hindi/tokenizer/train.py`

The Phase 1 training configuration uses an 80M-character training
budget and a target vocabulary size of 31,744.

See:

`hindi/train/README.md`

for the training procedure and reproducibility instructions.

## Evaluation

See:

`hindi/eval/README.md`

for the final Phase 1 corpus and tokenizer evaluation statistics.

## Corpus

The complete Hindi corpus is not stored in this repository because of
its size. It is provided separately through the project dataset storage.

See the root-level `README.md` for corpus access and project-wide
reproduction instructions.
