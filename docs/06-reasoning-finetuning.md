# Stage 6 — Reasoning fine-tuning and its evaluation

**Code:** `src/indic_lm/training/finetune.py`, `src/indic_lm/evaluation/reasoning.py`, `src/indic_lm/reasoning/` (data generation and checks)
**Data:** `data/reasoning/{hindi,nepali}/splits/test.jsonl` (held-out) · **Fresh results:** `artifacts/eval/`
**History:** `notebooks/03_phase3_finetune_kaggle.ipynb`, `reports/phase3/`

## What this stage does

Continues training the pretrained model on short question/answer text to see whether it can answer simple logic questions
(who is larger, smaller, equal, in what order), then measures the result on questions it did not train on.

```
templated questions (Hindi / Nepali) ─► dedup ─► split ─► fine-tune the base LM ─► held-out exact match vs baselines
```

Question categories in the Hindi held-out set (957 questions): `transitive_comparison` 228, `direct_comparison` 200, `ordering` 200,
`equality` 151, `multi_hop` 150, `basic_logic` 28. There are 22 distinct answer strings, of which "दोनों समान हैं" ("both are equal") is the most
common (153 questions, 16.0 %).

Two text formats:

```
qa    "प्रश्न: {question}\n\nउत्तर: {answer}"                              answer only
full  "प्रश्न: {question}\n\nसमाधान: {reasoning}\n\nउत्तर: {answer}"       reasoning, then answer
```

## Equations

**Fine-tuning loss** is the same next-token cross-entropy as pretraining, with padded positions excluded:

```
L = − Σ_t 1[y_t ≠ −100] · log p(y_t | y_<t)  /  Σ_t 1[y_t ≠ −100]
```

Hyper-parameters (unchanged from the Kaggle run): AdamW, lr 1e-5 constant, weight decay 0.01, batch 4, 3 epochs, clip 1.0; the best
validation-loss checkpoint is kept.

**Exact match** compares the text after the last `उत्तर:` on its first line with the gold answer, after NFC normalisation and
stripping punctuation (including the danda `।`). **Lenient match** accepts the gold answer as a whole word sequence inside the predicted line.

**Baselines** a model must beat before it can claim to have learned anything:

```
majority        = (count of the most frequent gold answer) / N                 always answer the same string
informed chance = (1/N) Σ_i 1[gold_i ∈ options_i] / |options_i|                pick uniformly among answer strings that literally appear in the question
                                                                              (a name-copier with no reasoning)
```

## Alternatives and why

| Decision | Chosen | Alternatives | Why / cost |
|---|---|---|---|
| Adaptation | Full fine-tuning of all 25 M weights | LoRA / adapters; prompting only | The model is tiny, so full fine-tuning is cheap. LoRA matters at billions of parameters |
| Loss target | **All tokens** (question + answer; only padding masked) | Loss on answer tokens only (prompt masking, standard in SFT) | Simplicity. Cost: much of the gradient teaches the model to reproduce templated *questions*, not to answer them. A likely reason for weak accuracy, **not tested** |
| Data | Synthetic, templated, verifiable | Human-written; translated benchmarks; LLM-generated | Cheap, exactly labelled, no pretrained model required. Cost: narrow. Questions share a small set of sentence templates and a name list |
| Format | answer-only and reasoning-then-answer both supported | One | Lets the same evaluator test with and without a reasoning trace |
| Learning rate | 1e-5, constant | Sweep; warmup + decay | Conservative, to avoid destroying pretraining. No sweep was run |
| Evaluation | Exact match against baselines, per category | Accuracy alone; LLM judge | Exact match is unambiguous for one-word answers and the baselines expose chance-level behaviour |
| Learning from feedback | Not used | RLHF, DPO, RL with verifiable rewards (the answers here *are* verifiable) | Out of scope; the natural next step for this task type |

## Results

### Hindi, held-out test set, answer-only prompt (`--prompt-mode qa`, 957 questions)

Checkpoint: `models/hindi_finetuned_phase3.pt` (the only fine-tuned model available for re-evaluation).

| | Exact match | Lenient |
|---|---:|---:|
| **Overall (85 / 957)** | **8.9 %** | 9.4 % |
| Majority-answer baseline | 16.0 % | — |
| Informed-chance baseline (copy a name from the question) | 24.5 % | — |

| Category | n | Exact match |
|---|---:|---:|
| transitive_comparison | 228 | 19.3 % |
| multi_hop | 150 | 17.3 % |
| ordering | 200 | 5.5 % |
| direct_comparison | 200 | 2.0 % (lenient 4.5 %) |
| equality | 151 | 0.0 % |
| basic_logic | 28 | 0.0 % |

Repetition rate is 0 % (the model no longer loops).

**What this says.**
1. The model scores **below the trivial "always say `दोनों समान हैं`" strategy (16.0 %)** and well below a name-copier (24.5 %). On this
   evidence it did **not** learn to reason, and its accuracy is not distinguishable from, or is worse than, guessing.
2. Only two categories reach ~17–19 %, roughly at the majority/chance level for those categories; `equality` and `basic_logic` are 0 %.
3. It answers in the right *format* (a short answer, one line, no looping), which is real but shallow progress.
4. An earlier 60-question pass with the same checkpoint gave 3.3 %; with 957 questions the number is 8.9 %. The gap between the two shows how
   unstable small samples are, and is the same reason the original report's "4/4 = 100 %" carries no evidential weight (95 % interval ≈ 40–100 %).

**What the wrong answers look like.** The output file keeps the first 20 questions it evaluated; all 20 happen to be wrong (a run of 20 misses is
unsurprising at 8.9 % accuracy). It is a sample, not a systematic study, but the breakdown is informative:

| Behaviour (n = 20) | Count | Example |
|---|---:|---|
| Names the wrong person on a chain question ("A > B, B > C: who is largest/smallest?") | 6 | five answer another person from the question, usually the middle of the chain; one names someone who is not in the question |
| Outputs the tie phrase "दोनों समान हैं" when the answer is a name | 4 | |
| Outputs a fragment of a *different* question template ("किसकी अंक अधिक है या क्या दोनों समान हैं?") instead of an answer | 6 | on "who has the most/least" questions listing three values |
| Answers an equality question with a name instead of "both equal" | 3 | |
| Garbled copy of the question preamble | 1 | |

A plausible entity with the wrong relation, template confusion, and a bias toward the most common answer are what a model without real multi-step
inference produces.

### Hindi, reasoning-first prompt (`--prompt-mode cot`, first 300 test questions, max 80 new tokens)

The prompt ends at `समाधान:`, so the model writes a reasoning trace and then `उत्तर:`. Only the first 300 questions of the test file were run (CPU time), so the
category mix is close to, but not identical with, the full set.

| | Exact match |
|---|---:|
| **Overall (99 / 300)** | **33.0 %** |
| Majority-answer baseline (same 300) | 17.3 % |
| Informed-chance baseline (same 300) | 24.4 % |
| **Overall excluding `equality` (49 / 250)** | **19.6 %** |

| Category | n | Exact match |
|---|---:|---:|
| equality | 50 | **100 %** |
| direct_comparison | 59 | 30.5 % |
| transitive_comparison | 66 | 22.7 % |
| ordering | 67 | 17.9 % |
| multi_hop | 53 | 7.5 % |
| basic_logic | 5 | 0 % |

**How to read this honestly.** The reasoning-first prompt beats both baselines overall, and it is far better than answer-only prompting (33.0 % vs 8.9 %, though on
different question subsets). But the headline is carried by `equality`: all 50 of those questions have the *same* answer string ("दोनों समान हैं"), and the model has learned to
emit it whenever a trace is requested. That is the majority-class strategy, not inference. Removing that category leaves 19.6 %, below the 24.4 % informed-chance
level, which was computed on all 300 questions and so is not the same subset, so treat that comparison as approximate.

The traces themselves show why the accuracy is not evidence of reasoning:

```
Q: प्रिया > नेहा, नेहा > कविता, कविता > मोहन. Who is smallest?   gold: मोहन
   trace: कविता > कविता, कविता > कविता, और कविता > कविता। इसलिए क्रम कविता > कविता > कविता > कविता > कविता है। सबसे छोटा कविता है।   answer: कविता   ✗

Q: कविता 90 किलोग्राम, मीना 50 किलोग्राम. Who weighs more?       gold: कविता
   trace: 80 > 50, इसलिए कविता की वजन अधिक है।                 answer: कविता   ✓ (the trace says 80 although the question says 90)
```

The model reproduces the *shape* of a worked solution (relations, "therefore", an ordering, an answer line) with the wrong content; in the second example the
answer is right while the trace misstates the number. A correct final answer is therefore not enough to claim a correct reasoning process, and the trace
cannot be trusted as an explanation.

### Nepali (60 questions, 10 per category, from the original Kaggle run)

| | Exact match | Repetition |
|---|---:|---:|
| Before fine-tuning | 0 / 60 (0 %) | 73.3 % |
| After fine-tuning | 6 / 60 (10 %) | 0 % |

All six correct answers are in `basic_logic` (6/10). The other five categories score 0/50. Generated text is very short for `multi_hop` (mean 12
characters) and `ordering` (1 character) against 50–101 expected, so part of that zero comes from truncation in the evaluation setup. **The Nepali
model could not be re-evaluated here** (checkpoint not available), so these are the saved numbers. See [known issues](known-issues.md), B1.

## Assessment

- **Genuine outcome:** fine-tuning turns a base model that babbles (73 % looping in Nepali) into one that emits a well-formed short answer or a
  well-formed worked solution. That is measurable and reproducible.
- **Partly achieved:** with a reasoning-first prompt the Hindi model is above the simple baselines overall (33.0 % vs 17.3 % / 24.4 %), driven by learning the constant
  answer for `equality`. Without it the score (19.6 %) is at or below chance level.
- **Not achieved:** reasoning. Answer-only accuracy is 8.9 % on the full test set (below both baselines), and the reasoning traces are largely degenerate
  (repeated names, wrong numbers), on templated questions where a system that reasons should approach 100 %.
- **Why, as hypotheses (none tested):** (a) the 25 M-parameter base model is too small and too weakly pretrained for multi-step
  inference; (b) the loss is spent on question tokens, not only on answers; (c) lr 1e-5 for 3 epochs on ~3 k examples is light; (d) the 128-token
  context and the tokenizer's space tokens ([stage 2](02-tokenizer.md)) leave little room for reasoning traces.
- **Experiments that would separate these:** mask the prompt in the loss; raise epochs/lr; run answer-only and reasoning-first fine-tuning for both
  languages on the same test set; fine-tune from the base checkpoints of both languages under identical settings; report per-category results against
  the baselines above; evaluate the full 957 questions in reasoning-first mode (only 300 were run here).

## Reproduce the evaluation

```bash
python -m indic_lm.evaluation.reasoning --language hindi --checkpoint models/hindi_finetuned_phase3.pt \
       --prompt-mode qa --out artifacts/eval/hindi_reasoning_qa.json
python -m indic_lm.evaluation.reasoning --language hindi --checkpoint models/hindi_finetuned_phase3.pt \
       --prompt-mode cot --max-new-tokens 80 --limit 300 --out artifacts/eval/hindi_reasoning_cot_first300.json
```

The output JSON holds overall and per-category exact/lenient match, repetition rate, both baselines, and 20 saved examples.
