# Documentation

One document per stage. Each states what was done and why, the equations, the alternatives with the reason for each choice,
real examples, and the limitations. Numbers are either computed from this repository or quoted from a saved report, and the source
is named next to them.

| # | Stage | Read it for |
|---|---|---|
| 1 | [Data pipeline](01-data-pipeline.md) | Cleaning, language filtering, exact and MinHash-LSH deduplication, document-level splits; Jaccard / MinHash / LSH / Bloom-filter equations |
| 2 | [Tokenizer](02-tokenizer.md) | Byte-level BPE from scratch, vocabulary-size study, cross-language cost, the "every space is a token" problem |
| 3 | [Model architecture](03-model-architecture.md) | Attention and block equations, the parameter budget (72 % in embeddings), alternatives such as RoPE, RMSNorm, SwiGLU |
| 4 | [Pretraining](04-pretraining.md) | AdamW and schedule equations, results, the code-vs-log learning-rate mismatch, resume equivalence |
| 5 | [Evaluation](05-evaluation.md) | Perplexity, bits/token, BLEU/chrF/ROUGE, Distinct-n, repetition; real generation examples with a good/bad verdict |
| 6 | [Reasoning fine-tuning](06-reasoning-finetuning.md) | Fine-tuning setup, held-out results against baselines, what the wrong answers look like |
| – | [Known issues](known-issues.md) | What must not be quoted, weak results, reproducibility gaps, design weaknesses |

Conventions: a **decision table** in each stage lists the choice made, the alternatives, why this one, and what it costs; **limitations**
close each document.
