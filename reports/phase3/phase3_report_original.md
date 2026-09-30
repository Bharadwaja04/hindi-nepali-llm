# Phase 3 Report — Hindi and Nepali Reasoning Fine-Tuning

## 1. Overview

Phase 3 focused on fine-tuning the language models for reasoning-oriented question answering in Hindi and Nepali.

The fine-tuning task used question–answer examples formatted as:

```text
प्रश्न: <question>

उत्तर: <answer>
```

The model was trained using causal language-model next-token prediction.

## 2. Model Configuration

| Parameter | Value |
|---|---:|
| Vocabulary size | 31,744 |
| Embedding dimension (`d_model`) | 288 |
| Attention heads | 8 |
| Transformer layers | 6 |
| Feed-forward dimension | 1,440 |
| Sequence length | 128 |
| Batch size | 4 |
| Epochs | 3 |
| Learning rate | `1e-5` |
| Weight decay | `0.01` |
| Optimizer | AdamW |
| Loss function | Cross-Entropy Loss |
| Padding/ignored label index | `-100` |

## 3. Dataset Preparation

The Hindi and Nepali datasets were loaded from JSONL files.

Each valid record required:

- a non-empty `question`
- a non-empty `answer`

The records were converted into instruction-style QA text using the language-specific question and answer format.

Separate training and validation splits were used for both languages.

## 4. Fine-Tuning Procedure

The following steps were performed:

1. Load the language-specific BPE tokenizer.
2. Load the pretrained language model checkpoint.
3. Convert the QA examples into tokenized training samples.
4. Create training and validation data loaders.
5. Fine-tune the model using AdamW.
6. Calculate training and validation loss after every epoch.
7. Save the checkpoint with the best validation loss.
8. Evaluate the fine-tuned model on reasoning questions.
9. Save the final model checkpoint and evaluation metrics.

## 5. Hindi Fine-Tuning

### Input and Checkpoints

- Base model: `hindi_model.pt`
- Tokenizer: `hindi_bpe_31744.json`
- Fine-tuned checkpoint: `hindi_qa_finetuned.pt`
- Final reasoning checkpoint: `hindi_qa_natural_comparison_final.pt`

### Evaluation

The Hindi evaluation included comparison-style reasoning questions involving relationships such as:

- greater than
- smaller than
- largest
- smallest

The notebook's final comparison evaluation reported:

```text
Comparison test accuracy: 4/4
```

This corresponds to:

```text
Accuracy: 100%
```

The complete Hindi metrics should be saved using the Hindi metrics snippet supplied with this report.

## 6. Nepali Fine-Tuning

### Input and Checkpoints

- Base model: `nepal_model.pt`
- Tokenizer: `nepali_bpe_31744.json`
- Training and validation data: Nepali JSONL splits
- Fine-tuned and final checkpoints: use the paths printed by the Nepali notebook cells

### Evaluation Metrics

The Nepali metrics were saved separately during the Kaggle run.

Fill the following values from the saved Nepali metrics file:

| Metric | Nepali result |
|---|---:|
| Number of test questions | `<insert value>` |
| Correct predictions | `<insert value>` |
| Accuracy | `<insert value>` |
| Accuracy percentage | `<insert value>%` |
| Final validation loss | `<insert value>` |

## 7. Metrics Files

The recommended metrics directory is:

```text
/kaggle/working/reasoning_project/metrics/
```

Expected files:

```text
hindi_metrics.json
hindi_metrics.txt
nepali_metrics.json
nepali_metrics.txt
```

The JSON files contain machine-readable evaluation results. The TXT files provide a human-readable summary of each question, expected answer, predicted answer, and correctness.

## 8. Limitations

- The evaluation set is small and should not be treated as a broad estimate of real-world reasoning ability.
- Exact-match accuracy is sensitive to formatting and spelling differences.
- A correct answer with additional text may be marked incorrect.
- The reported comparison accuracy reflects the selected test questions only.
- More diverse reasoning categories and a larger held-out test set are required for stronger evaluation.

## 9. Conclusion

Phase 3 produced fine-tuned Hindi and Nepali reasoning models using the project’s custom Transformer architecture and BPE tokenizers.

For Hindi, the final comparison evaluation achieved **4/4 correct predictions**, or **100% accuracy**, on the notebook’s selected test set.

The Nepali metrics were saved separately and should be inserted into Section 6 once the saved metrics file is opened.
