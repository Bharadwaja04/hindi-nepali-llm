# indic-lm — Hindi and Nepali language models, built from scratch

A complete, tested pipeline for training and evaluating small decoder-only language models for **Hindi** and **Nepali**, with no
pretrained weights, tokenizers or libraries beyond PyTorch: data cleaning and deduplication, a byte-level BPE tokenizer, a Transformer,
pretraining on ≈ 500 M tokens per language, evaluation, and a reasoning fine-tune.

It began as a university project and has been restructured as an installable package with tests, CI, a document per stage, and an
evaluation that reports **baselines and known problems, not only the good numbers**. Start with [`docs/`](docs/README.md).

```
raw text ─► clean / dedup / split ─► byte-level BPE ─► 25.3 M-param Transformer ─► pretrain (500 M tokens) ─► evaluate ─► reasoning fine-tune ─► evaluate vs baselines
            docs/01                   docs/02            docs/03                     docs/04                  docs/05      docs/06
```

## Results at a glance (all reproducible from this repository or its saved reports)

| Area | Result | Read it as |
|---|---|---|
| Model | 25,345,216 parameters (identical for both languages); 6 layers, d_model 288, 8 heads, context 128 | 72 % of the parameters are the embedding and output head |
| Tokenizer | 31,744 tokens per language; vocabulary study shows diminishing returns (Hindi fertility 2.73 → 2.56 → 2.44 → 2.37 at 8k / 16k / 32k / 65k) | The pre-tokenizer makes every space its own token (≈ 45 % of tokens), the largest design flaw ([docs/02](docs/02-tokenizer.md)) |
| Pretraining | 500 M tokens, 61,035 steps, ≈ 5 h (Hindi log); held-out perplexity **12.80** (Hindi), **13.53** (Nepali) | Perplexities are per token with different tokenizers and test sets, so they are **not comparable** across the two languages |
| Generation | Fluent news-style text; drifts off-topic, invents specifics, loops under greedy decoding | A working end-to-end model, not a useful assistant ([docs/05](docs/05-evaluation.md)) |
| Reasoning, Hindi (answer only) | **8.9 %** exact on the full 957-question held-out set, against a **16.0 %** always-the-same-answer baseline | Below baseline: no evidence of reasoning |
| Reasoning, Hindi (reasoning first) | 33.0 % on the first 300 questions, but 19.6 % once the constant-answer `equality` category is removed | Reasoning traces are degenerate ([docs/06](docs/06-reasoning-finetuning.md)) |
| Reasoning, Nepali | 0 / 60 → 6 / 60 exact; repetition 73 % → 0 % | The gain is stopping repetition; all six hits are in one category |
| Engineering | 33 automated tests (parameter count, causal-mask leak, resume equivalence, metric closed forms, …), ruff, GitHub Actions on Python 3.10–3.12 | |

The [known-issues](docs/known-issues.md) page lists what in the original coursework material should *not* be quoted (a 4-question "100 %" accuracy, an
impossible bits-per-byte value, a mislabelled best checkpoint) and what would be done differently.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # PyTorch, regex, numpy + pytest, ruff
make check                       # lint + 33 tests (CPU, seconds)
make tokenizer-check             # round-trip + fertility report on the released tokenizers
```

```python
from indic_lm.languages import get_language
from indic_lm.tokenizer import Tokenizer
from indic_lm.model import DecoderOnlyLM

spec = get_language("hindi")
tok = Tokenizer.load(spec.tokenizer_path)
model = DecoderOnlyLM.from_config(spec.load_model_config())     # random init; load a checkpoint to use trained weights
print(model.num_parameters())                                   # 25345216
ids = tok.encode("भारत एक सुंदर देश है।")                        # 10 tokens (4 are spaces)
assert tok.decode(ids) == "भारत एक सुंदर देश है।"
```

Train, fine-tune and evaluate (corpora and checkpoints are not in git, see [data/](data/README.md) and [models/](models/README.md)):

```bash
python -m indic_lm.training.pretrain --language hindi --manual-data ... --nonmanual-data ... --val-data ... --eval-every 1000
python -m indic_lm.training.finetune --language nepali --base-checkpoint models/nepali_pretrained.pt \
       --train data/reasoning/nepali/splits/train.jsonl --val data/reasoning/nepali/splits/validation.jsonl --format full
python -m indic_lm.evaluation.reasoning --language hindi --checkpoint models/hindi_finetuned_phase3.pt --prompt-mode qa
```

## Repository layout

```
src/indic_lm/           installable package
  languages.py            per-language settings in one place (no hindi/ and nepali/ copies of code)
  data/                   collect, clean, language filter, dedup (MinHash-LSH), splits, statistics, streaming batches
  tokenizer/              byte-level BPE: pre-tokenizer, trainer, encoder, decoder
  model/                  decoder-only Transformer, legacy-checkpoint loader
  training/               pretraining (resumable), reasoning fine-tuning, optimiser and schedule utilities
  evaluation/             perplexity / generation metrics; held-out reasoning evaluation with baselines
  reasoning/              synthetic reasoning-data generators and dataset verifiers
tests/                  33 tests, all CPU, seconds
configs/                model and training hyper-parameters per language
artifacts/              released tokenizers, evaluation outputs (small files only)
docs/                   one document per stage plus known issues
reports/                original phase reports and the raw metrics they cite (kept as history)
notebooks/              the original Colab / Kaggle notebooks (kept as history)
scripts/                tokenizer report
data/  models/          held-out reasoning test sets; checkpoints (gitignored)
```

## What this repository does not claim

- It is not a useful assistant, nor a benchmark result: at 25 M parameters and a 128-token context the models produce fluent but unreliable text.
- No comparison with other models was run, so nothing here says whether these perplexities are good relative to alternatives.
- Most design choices (learning rate, tied embeddings, RoPE, prompt masking) were not ablated. Only the tokenizer vocabulary size was measured.
- The reasoning data is templated; a high score on it would measure template matching, not reasoning.

## License

MIT. Data sources keep their own licences.
