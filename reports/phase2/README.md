# Phase 2 — Decoder-Only Causal Transformer

## 1. Overview

Phase 2 extends the Phase 1 Hindi and Nepali tokenizers into
decoder-only causal language models.

A separate model is trained for each language using its own
tokenizer and corpus.

The models are trained from scratch and are not initialized
from a pretrained GPT model.

---

## 2. Model Architecture

Both Hindi and Nepali use the same Transformer architecture.

| Component | Configuration |
|---|---:|
| Vocabulary size | 31,744 |
| Embedding dimension | 288 |
| Decoder blocks | 6 |
| Attention heads | 8 |
| Feed-forward dimension | 1,440 |
| Maximum sequence length | 128 |
| Dropout | 0.1 |
| Parameters | 25,345,216 |

### Architecture

The model consists of:

1. Learned token embeddings
2. Learned positional embeddings
3. Embedding dropout
4. Six decoder blocks
5. Final LayerNorm
6. Untied language-model head

Each decoder block uses:

- Pre-LayerNorm
- Causal multi-head self-attention
- Residual connection
- Feed-forward network
- GELU activation
- Dropout
- Residual connection

The attention mechanism uses an upper-triangular causal
mask so that a token cannot attend to future positions.

---

## 3. Causal Masking Verification

Causal masking was empirically tested for both languages.

Two identical input sequences were created, except for a
token at a future position.

The logits at positions before the modified token were then
compared.

The maximum difference was:

- Hindi: `0.0`
- Nepali: `0.0`

Therefore, changing a future token did not affect earlier
positions, confirming the causal masking implementation.

---

## 4. Training Configuration

| Setting | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.1 |
| Adam betas | (0.9, 0.95) |
| Warmup | 2,000 steps |
| Scheduler | Cosine decay |
| Minimum LR | 10% of initial LR |
| Gradient accumulation | 4 |
| Micro-batch size | 16 |
| Sequence length | 128 |
| Tokens / optimizer step | 8,192 |
| Gradient clipping | 1.0 |
| Target training tokens | 500M |
| Precision | FP16 autocast |

Training uses next-token prediction with cross-entropy loss.

---

## 5. Data

Large datasets and checkpoints are not stored in this repository.

The training data is kept externally and should be referenced
from the corresponding project storage.

The training pipeline supports:

- Manual corpus documents
- Non-manual corpus documents
- Training-split filtering
- Exact token budgeting
- Continuous token streaming
- Next-token batch construction

For Nepali, the verified training budget is:

- Manual corpus: 99,999,653 tokens
- Non-manual corpus: 400,000,000 tokens
- Total source budget: 499,999,653 tokens

The batching pipeline consumes 8,192 new source tokens per
optimizer step while maintaining continuity between batches.

---

# 6. Hindi Results

## 6.1 Training

The Hindi model completed:

- Steps: 61,035
- Training tokens: 499,998,720
- Training duration: approximately 4.98 hours

The best checkpoint during training occurred at step 6,000.

The final checkpoint was produced at step 61,000.

---

## 6.2 Final Evaluation

Final checkpoint:

`checkpoint_step_061000.pt`

| Metric | Result |
|---|---:|
| Validation loss | 2.549785 |
| Perplexity | 12.8044 |
| Bits/token | 3.678562 |
| BPB | 0.357112 |
| Evaluation tokens | 5,322,443 |

The final checkpoint is used for the primary reported evaluation.

The best intermediate checkpoint is also retained for comparison.

---

## 6.3 Text Generation

Generation was evaluated using:

- Greedy decoding
- Temperature 0.5
- Temperature 1.0
- Temperature 1.5

Example prompts included:

- `भारत में कृत्रिम बुद्धिमत्ता का भविष्य`
- `आज सुबह दिल्ली में मौसम`
- `शिक्षा के क्षेत्र में तकनीक`
- `भारत की अर्थव्यवस्था के लिए`
- `एक दिन एक छोटे से गाँव में`

The model showed more coherent output at lower temperatures,
while higher temperature increased variability and noise.

The model is a causal language model and is not instruction-tuned.

---

## 6.4 Generation Metrics

Evaluation used 100 validation examples.

The prompt consisted of the first 32 tokens and the reference
continuation contained the following 100 tokens.

| Metric | Result |
|---|---:|
| BLEU-4 | 0.1737 |
| chrF | 13.9414 |
| chrF++ | 11.9360 |
| ROUGE-L | 0.0025 |

> ROUGE-L is unusually low relative to the other metrics and
> should be sanity-checked before treating it as a final result.

---

### Evaluation Artifacts

The complete Hindi evaluation artifacts are available in:

- [Generation samples](hindi/generation_samples.json)
- [Generation evaluation results](hindi/generation_eval_results.json)
- [Generation evaluation set](hindi/generation_eval_set.json)
- [Generation metrics](hindi/generation_metrics.json)
- [Diversity metrics](hindi/diversity_metrics.json)
- [Training log](hindi/training_log.jsonl)
- [Attention statistics](hindi/attention/attention_statistics.json)

### Attention Visualizations

The following plots were generated during attention analysis:

- [Layer 1 attention](hindi/attention/attention_layer_1.png)
- [Layer 2 attention](hindi/attention/attention_layer_2.png)
- [Layer 3 attention](hindi/attention/attention_layer_3.png)
- [Layer 4 attention](hindi/attention/attention_layer_4.png)
- [Layer 5 attention](hindi/attention/attention_layer_5.png)
- [Layer 6 attention](hindi/attention/attention_layer_6.png)
- [Mean attention distance by layer](hindi/attention/mean_attention_distance_by_layer.png)
- [Attention entropy by layer](hindi/attention/attention_entropy_by_layer.png)
- [Attention distance by layer and head](hindi/attention/attention_distance_layer_head.png)

---

## 6.5 Diversity and Repetition

| Metric | Result |
|---|---:|
| Distinct-1 | 0.470002 |
| Distinct-2 | 0.911705 |
| 3-gram repetition rate | 0.000232 |

---

## 6.6 Attention Analysis

Attention was inspected across all six decoder layers.

The evaluation includes:

- Attention heatmaps
- Mean attention distance
- Attention entropy
- Layer/head attention statistics

The corresponding plots and statistics are stored with the
evaluation artifacts.

---

# 7. Nepali Results

## 7.1 Training

The Nepali model completed the full 500M-token training target.

- Steps: 61,035
- Training tokens: 499,998,720
- Parameters: 25,345,216

The final checkpoint is:

`final_model.pt`

The checkpoint is stored externally because of its size and is not committed
to GitHub.

---

## 7.2 Final Evaluation

The final checkpoint was evaluated on a deterministic held-out subset of
the Nepali test data using seed 42.

| Metric | Result |
|---|---:|
| Evaluation documents | 1,000 |
| Evaluation tokens | 359,396 |
| Loss | 2.604710 |
| Perplexity | 13.5273 |
| Bits/token | 3.757803 |

This is a held-out test subset evaluation rather than a complete evaluation
over the entire test corpus.

---

## 7.3 Text Generation

Generation was evaluated using:

- Greedy decoding
- Temperature 0.5
- Temperature 1.0
- Temperature 1.5

Example prompts included:

- `नेपाल`
- `नेपालको इतिहास`
- `नेपालको शिक्षा`
- `काठमाडौं`
- `नेपाली भाषा`
- `नेपालको संस्कृति`
- `हिमाल`
- `प्रविधि र विज्ञान`
- `आजको संसार`
- `नेपालमा विकास`

Lower temperatures generally produced more conservative and repetitive
outputs, while higher temperatures increased lexical diversity but reduced
reference similarity and coherence.

The model is a causal language model and is not instruction-tuned.

---

## 7.4 Generation Metrics

A deterministic reference-generation evaluation was performed using
20 held-out Nepali examples.

| Setting | BLEU-4 | chrF | chrF++ | ROUGE-L F1 |
|---|---:|---:|---:|---:|
| Greedy | 0.9720 | 19.1146 | 16.3334 | 0.1225 |
| Temperature 0.5 | 1.7575 | 19.9237 | 17.0897 | 0.1294 |
| Temperature 1.0 | 0.8023 | 19.4026 | 15.6178 | 0.0721 |
| Temperature 1.5 | 0.1543 | 17.5182 | 13.3287 | 0.0128 |

Temperature 0.5 produced the strongest reference-based scores among the
evaluated settings.

---

## 7.5 Diversity and Repetition

| Setting | Distinct-1 | Distinct-2 | 3-gram repetition |
|---|---:|---:|---:|
| Greedy | 0.185230 | 0.363971 | 0.401985 |
| Temperature 0.5 | 0.274818 | 0.556373 | 0.160050 |
| Temperature 1.0 | 0.419394 | 0.793865 | 0.052174 |
| Temperature 1.5 | 0.627986 | 0.957422 | 0.000000 |

Increasing temperature increased generation diversity and reduced repeated
3-grams. At temperature 1.5, diversity was highest but reference-based
generation scores were lowest.

---

## 7.6 Evaluation Artifacts

The complete final Nepali evaluation artifacts are available in:

- [Final validation metrics](nepali/final_validation_metrics.json)
- [Generation results](nepali/generation_results.json)
- [Diversity metrics](nepali/diversity_metrics.json)
- [Reference generation metrics](nepali/reference_generation_metrics.json)
- [Reference generation results](nepali/reference_generation_results.json)
- [Final evaluation report](nepali/report.md)

### Plots

- [Final validation metrics](nepali/plots/final_validation_metrics.png)
- [Generation diversity](nepali/plots/generation_diversity.png)
- [Generation repetition](nepali/plots/generation_repetition.png)
- [Reference generation metrics](nepali/plots/reference_generation_metrics.png)

### Attention Analysis

Attention analysis artifacts are also retained for the Nepali model:

- [Layer 1 attention](nepali/attention/attention_heatmap_layer_0.png)
- [Layer 2 attention](nepali/attention/attention_heatmap_layer_1.png)
- [Layer 3 attention](nepali/attention/attention_heatmap_layer_2.png)
- [Layer 4 attention](nepali/attention/attention_heatmap_layer_3.png)
- [Layer 5 attention](nepali/attention/attention_heatmap_layer_4.png)
- [Layer 6 attention](nepali/attention/attention_heatmap_layer_5.png)
- [Attention metrics](nepali/attention/attention_metrics.json)

---

# 8. Repository Structure

```text
common/
├── data.py
├── evaluation.py
├── training.py
└── tests

hindi/
├── configs/
├── model/
├── train/
└── eval/

nepali/
├── configs/
├── model/
├── train/
└── eval/

reports/
└── phase2/
    └── README.md