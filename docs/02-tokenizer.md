# Stage 2 — Tokenizer (byte-level BPE, trained from scratch)

**Code:** `src/indic_lm/tokenizer/` · **Artifacts:** `artifacts/tokenizers/{hindi,nepali}_bpe_31744.json`
**Experiments:** [`reports/phase1/phase1_report.md`](../reports/phase1/phase1_report.md) §11–14

## What this stage does

Turns text into integer IDs and back, with one tokenizer per language. Pipeline:

```
text ─► NFC normalise ─► regex pre-tokenise ─► UTF-8 bytes per piece ─► apply BPE merges ─► token IDs
```

Each language has its own vocabulary (31,744 = 256 byte tokens + 31,488 learned merges), trained only on that
language's corpus. No pretrained tokenizer or shared vocabulary is used.

| File | Role |
|---|---|
| `pretokenizer.py`, `config.py` | NFC + the regex that decides which pieces BPE may merge inside |
| `trainer.py` | BPE training on unique pre-tokens weighted by frequency; max-heap with lazy deletion |
| `encoder.py`, `decoder.py`, `vocab.py` | Apply merges by rank; concatenate token bytes and decode UTF-8 |
| `tokenizer.py` | Public `Tokenizer` class: `train`, `encode`, `decode`, `save`, `load`, `verify_round_trip` |
| `train.py`, `evaluate.py` | Command-line entry points |

## Equations

**BPE training.** Start from the 256 byte symbols. At each step, over the multiset of pre-tokens `w` with corpus
frequency `f(w)`, pick the most frequent adjacent pair and mint a new symbol:

```
(a*, b*) = argmax_(a,b)  Σ_w  f(w) · count_w(a, b)          new id = 256 + m   (m = 0 … 31,487)
```

Repeating this `M = 31,488` times gives `|V| = 256 + M = 31,744`. Working on *unique* pre-tokens weighted by
frequency, rather than every token occurrence, makes early merges cheap.

**Encoding** a piece: apply the applicable merge with the lowest rank (earliest learned) until none applies.
**Decoding**: concatenate each ID's bytes and decode UTF-8 (invalid sequences become `U+FFFD`).

**Metrics** used to judge a tokenizer:

```
fertility        F = (#tokens) / (#words)          lower is better
chars per token  = (#characters) / (#tokens)       higher is better
bytes per token  = (#UTF-8 bytes) / (#tokens)
```

## Decisions, alternatives, and why

| Decision | Chosen | Alternatives | Why this one | Cost |
|---|---|---|---|---|
| Algorithm | BPE | WordPiece; Unigram LM (SentencePiece); word-level; character-level | BPE is simple to implement from scratch, deterministic, and has a compact merge-list artifact. Implementing it was itself a course requirement | Unigram-LM tokenizers are often reported to align better with morphology in Indic languages; BPE is greedy and frequency-driven |
| Base alphabet | 256 **bytes** | Unicode characters with byte fallback (SentencePiece) | No unknown token is possible: emoji, Latin and digits all round-trip (unit-tested; `"AI मॉडल 2026 🙂"` → 8 tokens, lossless) | A Devanagari letter is 3 UTF-8 bytes, so the first merges are spent re-assembling letters (≈ 0.4 % of the merge budget) |
| Pre-tokenizer | `\s+ │ \p{L}[\p{M}\p{L}]* │ \p{N}+ │ [^\p{L}\p{N}\s]` | GPT-2 regex (attaches the leading space to a word); grapheme-cluster splitting; whitespace split | `\p{M}` keeps vowel signs and the virama inside the word, so merges never split a syllable from its matra; numbers group together | **Every space becomes its own token** — see the next section |
| Vocabulary size | 31,744 | 8k … 65k (all measured) | See the diminishing-returns table below; 31,744 = 496 × 64, close to the 32,768 point the experiment evaluated, and a multiple of 64 for GPU efficiency | Larger vocabularies shorten sequences but grow the embedding and output matrices |
| Separate vocab per language | One per language | One shared vocabulary | Course requirement; also better fit (measured below) | Cannot share subwords across the two languages |
| Special tokens | None defined | `<bos>`, `<eos>`, `<pad>`, `<unk>` | Not needed for plain next-token training on a concatenated stream | Documents are concatenated without a separator, and fine-tuning pads with ID 0 (the NUL byte). Reserving real special tokens is the cleanest fix |

## Evidence

### Vocabulary size vs fertility (Phase 1 experiment; 10,000 held-out documents per language)

| Vocab | Hindi fertility | Nepali fertility | Hindi chars/token | Nepali chars/token |
|---:|---:|---:|---:|---:|
| 8,192 | 2.7328 | 2.6989 | 2.06 | 2.20 |
| 16,384 | 2.5593 | 2.5446 | 2.20 | 2.33 |
| 24,576 | 2.4856 | 2.4801 | 2.27 | 2.39 |
| 32,768 | 2.4434 | 2.4417 | 2.31 | 2.43 |
| 49,152 | 2.3964 | 2.3975 | 2.35 | 2.48 |
| 65,536 | 2.3705 | 2.3717 | 2.38 | 2.50 |

Each doubling of the vocabulary buys less: 8k → 16k cuts Hindi fertility by 6.3 %, 16k → 32k by 4.5 %, and
32k → 65k by only 3.0 %. That is the evidence for stopping near 32k rather than going to 65k.

The same experiment split the 10,000 documents into five folds of 2,000 to check stability (report §12): the
downward trend holds in every fold; Hindi varies little (std ≈ 0.01) and Nepali more (std ≈ 0.07).

**Cost of a bigger vocabulary in *this* model.** With `d_model = 288`, the token embedding and the output layer
each hold `31,744 × 288 = 9.14 M` parameters. Together (18.3 M) that is **72 % of the model's 25.3 M
parameters**, so at this scale vocabulary size is a first-order architecture decision, not a detail.

### Why separate vocabularies (measured)

Encoding the first 400 held-out reasoning questions in each language with each tokenizer:

| Tokenizer | on Hindi text | on Nepali text |
|---|---:|---:|
| Hindi | **2.119** tokens/word | 2.684 |
| Nepali | 2.358 | **2.142** |

Using the other language's tokenizer costs about 25 % more tokens on Nepali text and 11 % more on Hindi text.
(Caveat: this text is templated question text, not corpus text; it shows direction, not exact corpus fertility.)

### Examples

```
Hindi   'भारत एक सुंदर देश है।'          55 bytes → 10 tokens: ['भारत', ' ', 'एक', ' ', 'सुंदर', ' ', 'देश', ' ', 'है', '।']
Nepali  'नेपालको राजधानी काठमाडौँ हो।'    78 bytes →  8 tokens: ['नेपालको', ' ', 'राजधानी', ' ', 'काठमाडौँ', ' ', 'हो', '।']
Hindi   'AI मॉडल 2026 🙂'                                 → 8 tokens, round-trips exactly
```

Common words become single tokens; rare words and other scripts fall back to smaller pieces, never to an
unknown token. Of the 31,488 learned merges, 20,939 decode to pure-Devanagari strings and the longest token
covers 82 characters (a frequent fixed phrase).

## The main weakness: spaces are tokens

In the examples above, 4 of 10 (Hindi) and 3 of 8 (Nepali) tokens are bare spaces. Measured over 400 questions
per language, **44.9 % (Hindi) and 44.3 % (Nepali) of all tokens are whitespace-only**; excluding them, fertility
is 1.17 and 1.19 tokens per word instead of 2.12 and 2.14.

Why it happens: the pre-tokenizer's `\s+` alternative makes whitespace its own piece, so BPE can never merge a
space into the following word. GPT-2-style tokenizers avoid this by matching ` ?\p{L}+`, which puts the space at
the start of the next word's piece.

Why it matters here: the model's context window is 128 tokens. At ~2.1 tokens per word, that is roughly 60 words;
at ~1.2 it would be roughly 105. Every downstream number (perplexity, generation length, how much reasoning fits in
one example) is affected. Changing the pre-tokenizer requires retraining the tokenizer *and* the models, so this is
recorded as the first item under "what I would change".

## Limitations

- Fertility numbers count spaces as tokens. Compare like with like: fertility is only comparable between
  tokenizers evaluated the same way (report §12 explicitly notes its stability-experiment values use a different
  formulation from the primary experiment).
- BPE training uses a streaming prefix of the corpus (up to the 80 M-character budget), so the vocabulary reflects
  whichever files come first in sorted order. Sampling uniformly across shards would remove that skew.
- `decode` on a *partial* byte sequence (as happens mid-generation) can yield `U+FFFD`.
