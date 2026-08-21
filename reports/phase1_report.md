# LMA Individual Project — Phase 1
## Data Collection and Tokenizer Construction
**Hindi + Nepali | Final End-to-End Report**

> **Status: Phase 1 complete.** This updated report incorporates the completed final split verification, tokenizer experiments, fertility stability analysis, final corpus construction, source provenance, artifact inventory, and GitHub submission checklist.

## 1. Phase 1 Configuration

| Item | Configuration |
|---|---|
| Higher-resource language | Hindi |
| Lower-resource language | Nepali |
| Tokenizer | BPE, trained from scratch |
| Final vocabulary | 31,744 per language |
| BPE merges | 31,488 per language |
| Final corpus target | ≈500M tokens/language |
| Manual-data requirement | ≥20% |
| Vocabulary experiment | 8,192 → 65,536 |
| Evaluation set | 10,000 documents/language |
| Phase 1 status | Complete |

## 2. Requirements and Scope

Phase 1 requires two independent monolingual pipelines, one higher-resource and one lower-resource. Each language has its own corpus, tokenizer, preprocessing pipeline, statistics, and train/validation/test splits.

The target is approximately 500M training tokens per language. At least 20% of final training tokens must come from manually collected data. Tokenizers must be trained from scratch, with no shared vocabulary or pretrained tokenizer.

## 3. Language Selection

### Hindi — Model H
Hindi was selected as the higher-resource language because of its substantial public corpus availability and broad written-domain coverage.

### Nepali — Model L
Nepali was selected as the lower-resource language from the assignment's allowed lower-resource language list. The supplied corpus documentation also identifies resource limitations and Devanagari orthographic variation.

## 4. Final Corpus Design

| Dimension | Hindi | Nepali |
|---|---:|---:|
| Language tier | Higher-resource | Lower-resource |
| Corpus type | Monolingual | Monolingual |
| Final training tokens | 499,999,998 | 499,999,956 |
| Manual tokens | 100,000,000 | 99,999,956 |
| Manual fraction | 20.00% | 20.00% |
| Non-manual tokens | 399,999,998 | 400,000,000 |
| Vocabulary | 31,744 | 31,744 |
| BPE merges | 31,488 | 31,488 |

Both languages meet the ≥20% manual-data requirement and are approximately at the 500M-token target.

## 5. Data Collection

### Hindi

The major non-manual source used in the final incremental expansion was **IndicCorp**. The final run processed 70 cleaned IndicCorp chunks.

**Downloaded / non-manual**
- IndicCorp
- AllenAI C4 Hindi
- Wikimedia/Wikipedia Hindi
- Hindi web-crawl batch

**Manual / extracted**
- Aaj Tak
- Hindi news material
- KavitaKosh
- Hindi RSS material
- Manually collected Hindi Wikipedia
- Hindi Wikisource
- Additional manual material

### Nepali

The final manual incremental stage explicitly used **Nepalipatra** and **KhabarHub**.

**Nepalipatra recorded statistics**
- Documents: 14,363
- Words: 2,623,251
- Characters: 17,216,800
- Bytes: 46,041,328
- Files: 3

**Other manual/extracted sources**
- Nepalipatra
- KhabarHub
- Kantipur
- Sahityaras
- CDC PDF extraction
- Internet Archive OCR
- National Library PDF extraction
- Additional manually extracted web material

**Downloaded / non-manual**
- `nepali_train.jsonl`
- `nepali_test.jsonl`

## 6. Cleaning and Normalization

The preprocessing pipeline includes:

- Unicode-preserving text processing
- Invalid/unwanted record removal
- Foreign-script contamination filtering where identified
- Duplicate removal
- Language-specific normalization
- Nepali orthographic handling
- Hindi validation/normalization/proofreading stages
- Independent Hindi and Nepali processing directories

## 7. Manual-Data Requirement

| Language | Manual | Non-manual | Final | Manual % |
|---|---:|---:|---:|---:|
| Hindi | 100,000,000 | 399,999,998 | 499,999,998 | 20.00% |
| Nepali | 99,999,956 | 400,000,000 | 499,999,956 | 20.00% |

The small shortfall from exactly 500M is caused by respecting complete document boundaries during incremental tokenization.

## 8. Tokenizer Construction

| Property | Hindi | Nepali |
|---|---|---|
| Algorithm | BPE | BPE |
| Vocabulary | 31,744 | 31,744 |
| BPE merges | 31,488 | 31,488 |
| Training | From scratch | From scratch |
| Shared vocabulary | No | No |
| Model file | `hindi_bpe_31744.json` | `nepali_bpe_31744.json` |

## 9. Tokenizer Sanity Tests

| Test | Hindi | Nepali |
|---|---:|---:|
| Encoded token count | 17 | 13 |
| Round-trip reconstruction | True | True |

Both tokenizers successfully reconstruct the tested text after encode/decode.

## 10. Final Incremental Tokenization

### Hindi
- Existing non-manual tokens: **67,722,646**
- Additional IndicCorp tokens: **332,277,352**
- Final non-manual: **399,999,998**
- Manual: **100,000,000**
- Final total: **499,999,998**
- IndicCorp chunks: **70**
- Documents: **2,643,291**
- Words: **151,813,130**
- Characters: **774,078,067**

The run stopped two tokens short of the incremental target because the next complete document would exceed the requested target.

### Nepali
- Existing manual tokens: **31,193,934**
- Additional manual tokens: **68,806,022**
- Final manual: **99,999,956**
- Non-manual: **400,000,000**
- Final total: **499,999,956**
- Files processed: **26** — 3 Nepalipatra + 23 KhabarHub
- Documents: **119,022**
- Words: **30,432,235**
- Characters: **200,620,657**

The run stopped 44 tokens short because the next complete document would exceed the target.

## 11. Vocabulary-Size Experiment

Fixed training/evaluation setup:
- Hindi training characters: **80,104,568**
- Nepali training characters: **80,065,007**
- Evaluation: **10,000 documents/language**

| Vocabulary | Hindi fertility | Nepali fertility | Hindi chars/token | Nepali chars/token |
|---:|---:|---:|---:|---:|
| 8,192 | 2.7328 | 2.6989 | 2.0622 | 2.1996 |
| 16,384 | 2.5593 | 2.5446 | 2.2020 | 2.3330 |
| 24,576 | 2.4856 | 2.4801 | 2.2673 | 2.3937 |
| 32,768 | 2.4434 | 2.4417 | 2.3064 | 2.4313 |
| 49,152 | 2.3964 | 2.3975 | 2.3517 | 2.4761 |
| 65,536 | 2.3705 | 2.3717 | 2.3774 | 2.5031 |

Fertility decreases monotonically as vocabulary size increases. Characters per token increase as vocabulary size grows.

## 12. Fertility Stability Experiment

The 10,000-document evaluation set was divided into five fixed subsets of 2,000 documents per language.

| Vocab | Hindi mean | Hindi std | Nepali mean | Nepali std |
|---:|---:|---:|---:|---:|
| 8,192 | 1.961455 | 0.016113 | 1.582750 | 0.066696 |
| 16,384 | 1.630386 | 0.013444 | 1.405148 | 0.076795 |
| 24,576 | 1.516787 | 0.012025 | 1.332388 | 0.077329 |
| 32,768 | 1.456364 | 0.010704 | 1.290962 | 0.075995 |
| 49,152 | 1.389938 | 0.009890 | 1.244039 | 0.072587 |
| 65,536 | 1.353215 | 0.009274 | 1.217993 | 0.069338 |

The decreasing fertility trend remains visible across fixed evaluation subsets. Hindi has relatively low between-subset variation. Nepali has higher variation but shows the same overall trend.

> **Important:** the absolute fertility values in the stability experiment are not numerically interchangeable with the primary vocabulary experiment because the two experiments use different evaluation formulations.

## 13. Interpretation

BPE learns frequent adjacent symbol sequences as merges. A larger vocabulary can represent longer reusable subword sequences directly, so fewer tokens are needed for the same text.

Therefore:
- fertility decreases;
- characters per token increase;
- early vocabulary increases give the largest gains;
- marginal gains become smaller at larger vocabularies.

## 14. Final Vocabulary Choice

The final vocabulary is **31,744** per language with **31,488 merges**. This is close to the experimentally evaluated 32,768 point, where the major early fertility gains are already captured, while avoiding progressively larger vocabulary sizes.

## 15. Final Train/Validation/Test Split Verification

### Hindi

| Split | Documents | Characters |
|---|---:|---:|
| Train | 7,061,250 | 2,152,245,131 |
| Validation | 71,912 | 21,537,503 |
| Test | 72,393 | 22,152,082 |

Hindi split verification completed successfully.

### Nepali

The Nepali split stage found **45,621 valid records** and produced an 80/10/10 split.

| Split | Documents | Characters |
|---|---:|---:|
| Train | 36,496 | 55,164,938 |
| Validation | 4,562 | 6,685,307 |
| Test | 4,563 | 6,979,674 |

Split artifacts:

```text
nepali/splits/train/nepali_train_00000.jsonl.gz
nepali/splits/validation/nepali_validation_00000.jsonl.gz
nepali/splits/test/nepali_test_00000.jsonl.gz
```

Statistics:

```text
phase1_report/nepali_split_statistics.json
```

These are document/character statistics, not token counts.

## 16. Artifact Inventory

| Artifact | Purpose |
|---|---|
| `hindi_bpe_31744.json` | Hindi BPE tokenizer/model |
| `nepali_bpe_31744.json` | Nepali BPE tokenizer/model |
| `vocabulary_fertility_results.csv` | Primary fertility results |
| `vocabulary_fertility_results.json` | Primary fertility results in JSON |
| `fertility_vs_vocab_size.png` | Primary fertility graph |
| `fertility_stability_results.csv` | Stability results |
| `fertility_stability_results.json` | Stability results in JSON |
| `fertility_stability_vs_vocab_size.png` | Stability graph |
| `fertility_comparison.png` | Hindi/Nepali comparison |
| `corpus_composition.png` | Corpus composition |
| `manual_data_percentage.png` | Manual-data percentage |
| `dataset_statistics.csv` | Dataset statistics |
| `phase1_metrics.csv` | Consolidated metrics |
| `phase1_metrics.json` | Consolidated metrics |
| `split_statistics.json` | Split verification |
| `nepali_split_statistics.json` | Nepali split statistics |
| `tokenization_examples.md` | Tokenization examples |
| `phase1_report.md` | Phase 1 repository report |

## 17. Assignment Deliverables Mapping

| Requirement | Status |
|---|---|
| Collection scripts | Complete |
| Preprocessing pipelines | Complete |
| Dataset statistics | Complete |
| Train/validation/test splits | Complete and verified |
| Tokenizer training code | Complete |
| Vocabulary/model files | Complete |
| Held-out vocabulary evaluation | Complete |
| Stability analysis | Complete |
| Tokenization examples | Complete |
| Graded figures | Complete |
| Phase 1 report | Complete |

## 18. Final Phase 1 Results

| Result | Hindi | Nepali |
|---|---:|---:|
| Final training tokens | 499,999,998 | 499,999,956 |
| Manual tokens | 100,000,000 | 99,999,956 |
| Manual fraction | 20.00% | 20.00% |
| Non-manual tokens | 399,999,998 | 400,000,000 |
| BPE vocabulary | 31,744 | 31,744 |
| BPE merges | 31,488 | 31,488 |
| Round-trip sanity | True | True |

## 19. GitHub Submission Checklist

- [x] Hindi collection/preprocessing code
- [x] Nepali collection/preprocessing code
- [x] Hindi tokenizer implementation
- [x] Nepali tokenizer implementation
- [x] Hindi/Nepali tokenizer model files
- [x] Dataset statistics
- [x] Final split statistics
- [x] Final split artifacts / required links
- [x] Vocabulary fertility CSV/JSON
- [x] Fertility graph
- [x] Stability CSV/JSON
- [x] Stability graph
- [x] Corpus composition figure
- [x] Manual-data percentage figure
- [x] Tokenization examples
- [x] `phase1_report.md`
- [x] External-storage/Drive documentation for large datasets
- [x] Graded figures stored in the repository rather than only in Drive

## 20. Reproducibility and Storage

Final corpus construction respects complete document boundaries. The small shortfalls from exactly 500M tokens are therefore expected and reproducible.

The vocabulary experiment uses a separate fixed 80M-character training/evaluation setup and must not be confused with the final 500M-token corpus.

Large corpus files should remain in the required external storage. GitHub should contain reproducibility code, statistics, tokenizer artifacts, reports and graded figures. The README/Phase 1 documentation should identify the corresponding Drive/external-storage locations.

## 21. Source Inventory — Manual vs Downloaded / Extracted

### Hindi — downloaded/non-manual
- IndicCorp
- AllenAI C4 Hindi
- Wikimedia/Wikipedia Hindi
- Hindi web-crawl batch

### Hindi — manual/extracted
- Aaj Tak
- Hindi news
- KavitaKosh
- Hindi RSS material
- Hindi Wikipedia
- Hindi Wikisource
- Additional manual material

### Nepali — downloaded/non-manual
- `nepali_train.jsonl`
- `nepali_test.jsonl`

### Nepali — manual/extracted
- Nepalipatra
- KhabarHub
- Kantipur
- Sahityaras
- CDC
- Internet Archive OCR
- National Library PDFs
- Additional extracted web material

### Final incremental runs

| Language | Source(s) | Role |
|---|---|---|
| Hindi | IndicCorp — 70 cleaned chunks | Non-manual expansion |
| Nepali | Nepalipatra — 3 chunks + KhabarHub — 23 chunks | Manual expansion |

The complete raw-source inventory is larger than the exact final token budget. The final incremental runs therefore need to be distinguished from the broader source pool.

## 22. Source Documents and Evidence

The report is grounded in:
1. LMA Individual Project specification.
2. Hindi Raw Text Corpus documentation.
3. Nepali Raw Text Corpus documentation.
4. Phase 1 execution logs.
5. Nepali source statistics.
6. Vocabulary fertility CSV/JSON and graph outputs.
7. Fertility stability CSV/JSON and graph outputs.
8. Final split verification statistics.
9. Project repository code and tokenizer/model artifacts.

## 23. Phase 1 Conclusion

**Phase 1 is complete.**

- Hindi and Nepali are independent monolingual pipelines.
- Final corpora are approximately 500M tokens each.
- Both meet the 20% manual-data requirement.
- Separate BPE tokenizers were trained from scratch.
- Each tokenizer has 31,744 vocabulary entries and 31,488 merges.
- Round-trip tokenizer sanity tests passed.
- Vocabulary-size evaluation demonstrates decreasing fertility with larger vocabularies.
- Stability analysis confirms the trend across fixed evaluation subsets.
- Final train/validation/test splits were generated and verified.
- Statistics, graphs, tokenizer artifacts and reports are prepared for repository submission.
- The resulting Phase 1 pipeline is ready to serve as the foundation for Phase 2.

## Appendix A — Key Repository Paths

```text
hindi/data/
hindi/tokenizer/
hindi/model/hindi_bpe_31744.json

nepali/data/
nepali/tokenizer/
nepali/model/nepali_bpe_31744.json

nepali/splits/train/nepali_train_00000.jsonl.gz
nepali/splits/validation/nepali_validation_00000.jsonl.gz
nepali/splits/test/nepali_test_00000.jsonl.gz

reports/phase1_report.md
scripts/phase1_pipeline.ipynb
scripts/bootstrap_colab.py
scripts/scrape_manual.py
```

Large raw/processed corpora remain in external project storage and are documented through the repository README.
