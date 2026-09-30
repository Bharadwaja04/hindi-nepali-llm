# Stage 5 — Evaluation of the pretrained models

**Code:** `src/indic_lm/evaluation/metrics.py` · **Saved outputs:** `reports/phase2/{hindi,nepali}/`
**Tests:** `tests/test_pipeline_units.py` (closed-form checks of perplexity, bits/token, Distinct-n and repetition rate; BLEU/chrF/ROUGE were computed by the original evaluation scripts and are not re-tested here)

## What this stage does

Answers "is the base model any good?" from three angles: how well it predicts held-out text (intrinsic), what it writes
when asked to continue a prompt (extrinsic, generation), and whether that writing is varied or degenerate.

| Question | Metric | Needs a reference? |
|---|---|---|
| How surprised is the model by real text? | loss, perplexity, bits/token | no (held-out text) |
| Does generated text resemble the real continuation? | BLEU-4, chrF, chrF++, ROUGE-L | yes |
| Is the generated text varied or looping? | Distinct-n, n-gram repetition rate | no |
| Does the model use context sensibly? | attention entropy / distance per head (artifacts under `reports/phase2/*/attention`) | no |
| Does it answer questions correctly? | exact-match against baselines | yes: see [stage 6](06-reasoning-finetuning.md) |

## Equations

```
loss (nats/token)  L = −(1/N) Σ_t log p(x_t | x_<t)
perplexity         PPL = exp(L)                      ("effective branching factor")
bits per token     L / ln 2  =  log2(PPL)
bits per byte      BPB = (N_tokens · L / ln 2) / N_bytes  =  (bits/token) / (bytes/token)
```

BPB is the metric that *can* be compared across tokenizers, because the denominator is the raw text and not the token
count. Perplexity and bits/token cannot.

```
Distinct-n       = |unique n-grams| / |n-grams|                    higher = more varied
repetition rate  = fraction of n-grams (n = 3) that occurred earlier in the same text
BLEU-4           = BP · exp( ¼ Σ_{n=1..4} log p_n )                p_n = modified n-gram precision
chrF             = character n-gram F-score (β = 2, recall-weighted); chrF++ adds word 1–2-grams
ROUGE-L          = F-score of the longest common subsequence
```

## Alternatives and why these

| Choice | Chosen | Alternatives | Reason and cost |
|---|---|---|---|
| Intrinsic metric | Perplexity, bits/token | BPB; downstream benchmarks (IndicGLUE, IndicXNLI, MMLU-style suites) | Cheapest, label-free and directly what the loss optimises. Does not tell you if the model is *useful*, and it is tied to the tokenizer. **Benchmarks were not run** because a 25 M-parameter model trained on news is far below what they measure |
| Generation similarity | BLEU, chrF, ROUGE-L | BERTScore (needs a pretrained encoder, not allowed); human rating; LLM-as-judge | Overlap metrics are computable without any pretrained model. Weakness: an open-ended continuation has many valid answers, so a *good* continuation can score near zero. Treat these as relative, between decoding settings, not absolute |
| Character-level metric | chrF | word-level BLEU | chrF is more forgiving of Hindi/Nepali inflection and does not depend on word segmentation |
| Diversity | Distinct-n, repetition | self-BLEU, MAUVE | Simple and cheap. Distinct-n alone rewards random noise (see the temperature-1.5 row below), so it must be read together with the similarity scores |
| Decoding settings | greedy, T = 0.5, 1.0, 1.5 | top-k, nucleus (top-p), beam search, repetition penalty | Bracket the range from deterministic to noise. Nucleus sampling and a repetition penalty are the usual fix for the looping seen below and were not tried |

## Results

### Intrinsic (held-out, per-token)

| | Hindi | Nepali |
|---|---:|---:|
| Validation loss | 2.5498 | 2.6047 |
| Perplexity | 12.80 | 13.53 |
| Bits/token | 3.679 | 3.758 |
| Tokens evaluated | 5,322,443 | 359,396 (1,000 docs, seed 42) |

Not comparable across languages (different tokenizers and different test sets). See [stage 4](04-pretraining.md).
Do **not** quote the reported Hindi BPB (0.357): it contradicts bits/token ([known issues](known-issues.md), A2).

### Generation, Nepali (20 held-out prompts: 32-token prompt, 100-token reference)

| Setting | BLEU-4 | chrF | chrF++ | ROUGE-L | Distinct-1 | Distinct-2 | 3-gram repetition |
|---|---:|---:|---:|---:|---:|---:|---:|
| Greedy | 0.97 | 19.11 | 16.33 | 0.123 | 0.185 | 0.364 | **0.402** |
| T = 0.5 | **1.76** | **19.92** | **17.09** | **0.129** | 0.275 | 0.556 | 0.160 |
| T = 1.0 | 0.80 | 19.40 | 15.62 | 0.072 | 0.419 | 0.794 | 0.052 |
| T = 1.5 | 0.15 | 17.52 | 13.33 | 0.013 | 0.628 | 0.957 | 0.000 |

### Generation, Hindi (100 validation examples, greedy)

BLEU-4 0.17 · chrF 13.94 · chrF++ 11.94 · ROUGE-L 0.0025 (the last is suspect, see known issues A4).

**Reading the tables.** The trade-off is textbook. Greedy decoding gets stuck (40 % of 3-grams repeat). Temperature
1.5 has the best diversity (0.957) and the worst similarity: high Distinct-n there is noise, not creativity.
T = 0.5 is the best compromise on the Nepali similarity scores. BLEU-4 below 2 means exact 4-gram overlap with the
reference is almost nil, which is normal for free continuation of a news snippet and says little about quality by itself.

## Examples and honest assessment

**Hindi, prompt** `भारत में कृत्रिम बुद्धिमत्ता का भविष्य` ("the future of artificial intelligence in India").

| Setting | Output (start) | Verdict |
|---|---|---|
| Greedy | "…का भविष्य है। भारत में इस साल के अंत तक भारत में सबसे ज्यादा कोरोना वायरस संक्रमण के मामले सामने आए हैं। भारत में कोरोना वायरस के कुल मामले बढ़कर 1,913 हो गए हैं। वहीं देश में कोरोना वायरस के कुल मामले बढ़कर 1,913 हो गए हैं।" | **Fluent but wrong topic.** Grammatical Hindi in news register, yet it ignores "AI" and drifts to COVID case counts (the corpus is news-heavy), invents a number, and repeats a whole sentence. |
| T = 0.5 | Same drift, more variety, still COVID | Fluent, off-topic |
| T = 1.0 | Grammatical fragments, a garbled token (`ौना`), unrelated subjects | Degrading |
| T = 1.5 | A run of unrelated words and Latin fragments, no sentences | Noise |

**Nepali, prompt** `नेपालको शिक्षा` ("education in Nepal"), greedy: "नेपालको शिक्षा क्षेत्रमा महत्वपूर्ण योगदान … उनले भने,
'नेपालको शिक्षा क्षेत्रमा लगानी गर्न आवश्यक छ । तर यो क्षेत्रमा लगानी गर्न आवश्यक छ । तर यो क्षेत्रमा लगानी गर्न आवश्यक …'". It begins
in convincing news style, stays on topic for a sentence, then loops on one clause. Another prompt (`नेपालको इतिहास`) loops on
"राष्ट्रिय एकता कायम गर्न".

**Assessment, plainly.**
- *Good:* the model learned the script, spelling, punctuation, quotation and register of news text from scratch, and short-range
  grammar is mostly right. Both languages produce plausible sentences.
- *Bad:* no reliable topical control (Hindi drifts from AI to COVID), it fabricates specifics (numbers, and in some Nepali
  outputs, real named people in invented roles, which is why they are not reproduced here), it repeats under greedy
  decoding, and higher temperature buys variety only by giving up coherence.
- *This is what a 25 M-parameter, 128-token-context, single-domain model should look like.* It is a working end-to-end
  pipeline, not a useful assistant. It must not be used to produce factual content.

## Limitations of the evaluation itself

- Sample sizes are small for generation (Nepali: 20 prompts; Hindi: 100), with no confidence intervals or multiple seeds. The
  difference between T = 0 and T = 0.5 on 20 examples is suggestive, not established.
- All test text comes from the same distribution as training (news-heavy web text), so results say nothing about other domains.
- There is no downstream or human evaluation, and no comparison with any other model at the same scale (for example a
  multilingual pretrained model of similar size), so there is nothing to say whether the 12.8 perplexity is good *relative* to alternatives.
- Nepali intrinsic numbers come from a 1,000-document subset of the test split, not the full split.
