# Hindi Tokenizer Training

This directory documents the Phase 1 training configuration for the
Hindi BPE tokenizer.

The actual tokenizer training implementation is located at:

`hindi/tokenizer/train.py`

The trained tokenizer artifact is stored at:

`hindi/model/hindi_bpe_31744.json`

---

## Training Configuration

| Parameter | Value |
|---|---:|
| Language | Hindi |
| Vocabulary size | 31,744 |
| BPE merges | 31,488 |
| BPE training character budget | 80,000,000 |

---

## Training Data

The final Hindi Phase 1 corpus contains:

- 500,000,000 tokens
- 100,000,000 manual tokens
- 400,000,000 non-manual tokens

The complete corpus is not stored in this GitHub repository because of
its size.

The BPE tokenizer itself is trained using an 80M-character sample of
the corpus. The full 500M-token corpus is used for the final
tokenization stage.

Therefore:

```text
500M-token corpus
       |
       +----> 80M-character training sample
       |             |
       |             v
       |        BPE training
       |             |
       |             v
       |        31,744 vocabulary
       |        31,488 merges
       |
       +----> final corpus tokenization
```

---

## Training Command

From the repository root:

```bash
python -m hindi.tokenizer.train \
    --language hindi \
    --data <PATH_TO_HINDI_TRAINING_DATA> \
    --vocab-size 31744 \
    --output hindi/model/hindi_bpe_31744.json \
    --char-budget 80000000
```

---

## Training Procedure

The tokenizer training process is:

1. Load the Hindi training text.
2. Normalize the text using Unicode NFC normalization.
3. Apply the Unicode-aware regex pre-tokenizer.
4. Represent text using UTF-8 bytes.
5. Count adjacent symbol-pair frequencies.
6. Merge the most frequent eligible pair.
7. Repeat until the target vocabulary size is reached.
8. Save the learned vocabulary and merge operations.

The tokenizer is trained independently from the Nepali tokenizer.

Hindi and Nepali do not share their learned merge tables or
vocabularies.

---

## Output

The final trained tokenizer is:

`hindi/model/hindi_bpe_31744.json`

Its configuration is:

```text
Language        : Hindi
Vocabulary size : 31,744
BPE merges      : 31,488
Training budget : 80,000,000 characters
```

---

## Reproducibility

The tokenizer implementation required to reproduce training is included
in:

`hindi/tokenizer/`

The large corpus is provided separately through the project dataset
storage and is not included in GitHub.
