# Known issues and honest limitations

Found while restructuring the original coursework repository. Each item states what is wrong, the evidence, the
impact, and the fix. None of these are hidden in the numbers elsewhere in the docs. Items are ordered by how much they
change what you should believe about the project.

## A. Claims that must not be repeated

| # | Claim in the original material | Problem | Evidence |
|---|---|---|---|
| A1 | "Hindi reasoning accuracy: **4/4 = 100 %**" (Phase 3 report) | The test set has **four questions**. One question is 25 percentage points; the 95 % confidence interval for 4/4 is roughly 40 %–100 %. It is a demo, not a metric. | `reports/phase3/phase3_report_original.md` §5 |
| A2 | Hindi BPB = 0.357 (Phase 2 report) | Inconsistent with the same report's 3.679 bits/token. It implies 10.3 bytes/token, but a Devanagari character is 3 bytes and the tokenizer yields ≈ 2.3 characters/token, so ≤ ≈ 7 bytes/token. | [04-pretraining](04-pretraining.md), issue 5 |
| A3 | "Best Hindi checkpoint at step 6,000" | Training loss at steps 1,001–6,000 averages 4.05 vs 2.56 at the end; step 6,000 cannot be best under any loss-based rule. | training log |
| A4 | Hindi ROUGE-L 0.0025 | The report itself flags this as suspicious: 0.0025 next to chrF ≈ 14 and BLEU-4 0.17 is hard to believe (the Nepali run reports ROUGE-L 0.12 on the same kind of task). The saved metrics do not say how text was tokenised for ROUGE, so the cause is unconfirmed. Treat as unverified until recomputed. | `reports/phase2/README.md` §6.4 |

## B. Results that are real but weak

| # | Finding | Numbers |
|---|---|---|
| B1 | Nepali reasoning fine-tune fixed the format, but correctness is concentrated in one category and the evaluation truncates output | Before: **0/60** exact, 73.3 % of outputs degenerate into repetition (`म त म त म त …`). After: **6/60 (10 %)**, 0 % repetition. Recomputed from the saved `after_finetuning_results.json` (10 questions per category): **all 6 correct answers are in `basic_logic` (6/10); the other five categories score 0/50.** Mean generated length is 63 characters for `basic_logic` but only **12 for `multi_hop` and 1 for `ordering`** (expected reasoning: 101 and 50 characters), so those outputs were cut off or empty. The saved files do not say whether the cap was on new tokens or on total length, so those zeros are partly an evaluation artifact and cannot be read as "the model cannot do it". A real gain is stopping the repetition; the accuracy figure itself is not a reliable measure of reasoning. |
| B2 | Reasoning data is **templated** | The questions are generated from a small set of sentence templates with a name list (e.g. "if A's value exceeds B's and B's exceeds C's, who is smallest?"). Held-out questions share templates with training questions, so even a high score would measure template matching, not reasoning. The Hindi generator and Nepali generator differ, so the two languages are not tested on the same task. |
| B3 | Perplexity ≈ 13 at 25 M parameters | Fluent-looking short text, weak factual grounding. Greedy decoding loops (Nepali greedy: 40 % of 3-grams are repeats, see `reports/phase2`). |
| B4 | Held-out Hindi reasoning result (recomputed for this repository) | Answer-only, full 957-question test set: **8.9 %** exact, below the majority baseline (16.0 %) and informed chance (24.5 %). Reasoning-first, first 300 questions: 33.0 % overall, but the `equality` category (always the same answer) is 50/50; without it 19.6 %. Reasoning traces are degenerate (repeated names, wrong numbers). See [06-reasoning-finetuning](06-reasoning-finetuning.md). |

## C. Reproducibility gaps

| # | Gap | Impact | Fix |
|---|---|---|---|
| C1 | Learning-rate schedule in the released Hindi log (cosine to **0**) differs from the code (cosine to **10 %** floor). Verified by simulating the code for 61,035 steps. | A rerun will not reproduce the exact curve. | Choose one; record it in the config. |
| C2 | Weight decay applies to biases, LayerNorm and embeddings. | Non-standard; effect unmeasured. | Use two parameter groups (decay only ≥ 2-D weights) and ablate. |
| C3 | No validation loss during original pretraining. | Cannot plot the train/validation gap over time. | `pretrain.py --eval-every` now supports it; not applied to the released runs. |
| C4 | Corpora and pretrained checkpoints are not in git. | Cannot rerun training or evaluate the base models from the repo alone. | Data: `data/README.md` (external storage). Weights: attach to a GitHub Release. |
| C5 | Only the fine-tuned **Hindi** checkpoint was available for re-evaluation; base checkpoints and the Nepali fine-tuned checkpoint were not. | Nepali fine-tune results are taken from the saved Kaggle metrics files, not recomputed. | Supply `nepal_model.pt`, `hindi_model.pt` to rerun. |
| C6 | Corpus size figures differ between reports because they describe different stages (raw, cleaned, deduplicated, tokenised). | Confusing when comparing. | Generate one `corpus_statistics.json` from `data/stats.py` and cite only that. |
| C7 | Original scripts hard-coded `/kaggle/...` and Colab paths, and the fine-tuning code existed only as raw notebook cell dumps. | Not runnable outside those machines. | Refactored into `indic_lm.training.finetune`; notebooks kept as history in `notebooks/`. |

## D. Design weaknesses (a different design would be better)

| # | Weakness | Why it matters | What to do |
|---|---|---|---|
| D1 | The pre-tokenizer makes **every space a token** (≈ 45 % of all tokens). | Halves the effective context: 128 tokens ≈ 60 words instead of ≈ 105. Inflates perplexity's token count. | Attach the space to the following word in the regex; retrain tokenizer and models ([02-tokenizer](02-tokenizer.md)). |
| D2 | No special tokens (`<pad>`, `<eos>`, `<bos>`). | Documents run together in the stream; fine-tuning pads with ID 0 (the NUL byte). | Reserve IDs at the start of the vocabulary. |
| D3 | Language ID by script ratio cannot tell Hindi from Nepali. | Possible cross-language contamination in both corpora. | Train a small char-n-gram classifier from scratch on seed text ([01-data-pipeline](01-data-pipeline.md)). |
| D4 | 72 % of parameters are embedding and output head. | Little capacity in the blocks. | Tie the embeddings and/or shrink the vocabulary, then test ([03-model-architecture](03-model-architecture.md)). |
| D5 | No ablations of any architecture or training choice. | The "why" behind those choices is convention, not measurement. | The tokenizer vocabulary-size study is the only measured design decision; add LR sweep, tied vs untied, RoPE vs learned. |
| D6 | Evaluation is in-domain only (a random document split of a news-heavy corpus). | Says nothing about other domains. | Add a held-out-source split (train on some sites, test on others). |

## What is genuinely solid

- The from-scratch tokenizer, model and training loop run end-to-end and are covered by 33 automated tests, including a
  parameter-count check (25,345,216), causal-mask leak test, checkpoint round-trip and resume-equivalence.
- The reported perplexities are internally consistent with the reported losses and bits/token.
- The data pipeline (cleaning, exact and MinHash dedup, document-level splits) is implemented with the equations in the docs.
- The vocabulary-size study is a real measurement with a diminishing-returns result.
