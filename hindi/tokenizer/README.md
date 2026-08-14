# Language Tokenizer

This folder contains the from-scratch regex-pre-tokenized byte-level
BPE tokenizer used by the Hindi/Nepali language-model project.

## Architecture

```text
text
  -> Unicode NFC
  -> regex pre-tokenization
  -> UTF-8 bytes
  -> BPE
  -> token IDs
```

No GPT-4/tiktoken vocabulary or pretrained merge table is used.

## Files

- `tokenizer.py` - public tokenizer interface
- `bpe.py` - core BPE operations
- `pretokenizer.py` - Unicode-aware regex pre-tokenization
- `trainer.py` - BPE training
- `encoder.py` - text -> token IDs
- `decoder.py` - token IDs -> text
- `vocab.py` - vocabulary construction
- `config.py` - tokenizer configuration
- `train.py` - command-line training entry point
- `evaluate.py` - evaluation entry point

## Install

```bash
pip install regex
```

## Train

From the repository root:

```bash
python -m hindi.tokeniser.train     --language hindi     --data data/clean/hindi     --vocab-size 32000     --output hindi/tokeniser/hindi_tokenizer.json
```

For Nepali:

```bash
python -m nepali.tokeniser.train     --language nepali     --data data/clean/nepali     --vocab-size 32000     --output nepali/tokeniser/nepali_tokenizer.json
```

## Important

Hindi and Nepali must be trained separately. Do not share their
merge tables or vocabulary.
