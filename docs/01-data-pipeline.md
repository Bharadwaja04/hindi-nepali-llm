# Stage 1 — Data pipeline

**Code:** `src/indic_lm/data/` (shared) · `src/indic_lm/data/hindi/` (Hindi-specific variants)
**Report with corpus numbers:** [`reports/phase1/phase1_report.md`](../reports/phase1/phase1_report.md)

## What this stage does

```
raw sources ─► collect ─► clean ─► language filter ─► exact dedup ─► near-dup (MinHash) ─► split ─► stats
```

| Step | Module | What it does |
|---|---|---|
| Collect | `data/collect.py` | Streams public Hindi/Nepali corpora (mC4, OSCAR, Wikipedia, IndicCorp) from Hugging Face into JSONL shards tagged `source: downloaded`. Manually collected material (news sites, Wikisource, PDFs/OCR) is tagged `source: manual`. |
| Clean | `data/clean.py` | NFC-normalise, strip control characters, HTML tags, markdown links, URLs and e-mail addresses, collapse runs of repeated characters/words and whitespace, then apply length, script and repetition filters. |
| Language filter | `data/langfilter.py` | Keeps a document only if ≥ 55 % of its non-space characters are Devanagari. |
| Exact dedup | inside `clean.py` | SHA-1 of the cleaned text, checked against a Bloom filter plus a bounded exact-hash set. |
| Near-dup | `data/dedup.py` | MinHash + LSH over 5-word shingles. |
| Split | `data/splits.py` | Seeded, document-level train/validation/test split via a streaming reservoir shuffle. |
| Stats | `data/stats.py` | Documents, characters, bytes, manual-vs-downloaded breakdown, tokens per character/byte with a trained tokenizer. |

Hindi additionally has its own scripts in `data/hindi/` (a different preprocessing/language-filter pair); the Nepali pipeline uses the shared modules directly — the per-language Nepali copies in the original repo were byte-identical to them.

**Corpus budget (Phase 1 report):** ≈ 500 M training tokens per language, of which 20 % (100 M) are manually collected. Hindi finished at 499,999,998 tokens and Nepali at 499,999,956; the tiny shortfall exists because tokenization stops at a complete-document boundary.

## Equations

Near-duplicate detection treats each document as a set of word 5-grams (shingles) `S`. Similarity is Jaccard:

```
J(A, B) = |S_A ∩ S_B| / |S_A ∪ S_B|
```

MinHash makes this cheap. For a random hash `h`, `P[ min h(S_A) = min h(S_B) ] = J(A, B)`, so with `k` hash functions the estimate is

```
Ĵ(A, B) = (1/k) · Σ_{i=1..k} 1[ min h_i(S_A) = min h_i(S_B) ]        (k = 128 here)
```

LSH avoids comparing every pair: the 128-value signature is cut into `b` bands of `r = k/b` rows and two documents become candidates if any band matches exactly. With `b = 16`, `r = 8`:

```
P[candidate | J = s] = 1 − (1 − s^r)^b          approximate threshold ≈ (1/b)^(1/r) ≈ 0.71
```

Candidates are then verified with `Ĵ ≥ 0.9` before a document is dropped.

Exact-dedup false positives from the Bloom filter: with `n` items, `m` bits and `k` hashes, `p ≈ (1 − e^(−kn/m))^k`; the filter is sized for `p = 10⁻⁴`, i.e. about 1 in 10,000 *unique* documents may be wrongly dropped.

## Decisions, alternatives, and why

| Decision | Chosen | Alternatives | Why this one | Cost / what it gives up |
|---|---|---|---|---|
| Quality filtering | Hand-written heuristics (length 80–100 000 chars, script ratio, unique-word ratio ≥ 0.65, strip boilerplate) | Model-based filters (KenLM perplexity, a classifier trained to prefer Wikipedia-like text); C4/Gopher-style rule sets (terminal punctuation, line-length, bad-word lists); dedicated extractors such as trafilatura | The course rules forbid pretrained artefacts, and heuristics are deterministic, streamable and cheap on a 22 GB input | Blunt: the repetition filter can reject legitimately repetitive text (poetry, lists); no PII scrubbing |
| Unicode form | NFC | NFKC, NFD, none | NFC is canonical (no meaning change) and makes visually identical Devanagari strings identical: e.g. `क़` (U+0958) and its decomposed form encode to the same tokens (unit-tested) | NFKC would also fold compatibility characters, which is riskier for a language where variant forms carry meaning |
| Language ID | Devanagari-character ratio ≥ 0.55 | fastText `lid.176`, CLD3 (pretrained → not allowed); n-gram classifier trained on labelled seed text; stop-word lists | No pretrained model needed; catches Latin/other-script contamination | **Cannot tell Hindi from Nepali** (same script), nor from Marathi or Sanskrit. Cross-contamination between the two corpora is possible |
| Exact dedup | SHA-1 + Bloom filter + bounded set | Full in-memory hash set; sort-based dedup; external DB | Bounded RAM on a Colab-class machine | ~0.01 % of unique documents lost to Bloom false positives; after 200 k entries the exact set is cleared and only the filter remembers |
| Near-dup | MinHash-LSH, 128 permutations, 5-word shingles, threshold 0.9 | SimHash; exact-substring dedup with suffix arrays (Lee et al. 2022); embedding-based semantic dedup; the `datasketch` library | Standard for web-scale corpora; catches boilerplate replays and lightly edited copies | Pure-Python hashing is O(k · shingles) per document — slow at millions of documents; vectorising with NumPy or using `datasketch` would be the first optimisation. Threshold 0.9 keeps near-copies with substantial edits |
| Split | Document-level, seeded reservoir shuffle, default 98/1/1 | Sentence-level (leaks), split by source or date (tests domain shift), stratified | Document-level avoids leakage across splits; a seed makes it reproducible | A random split mirrors the training distribution, so it measures in-domain loss only (news-heavy) |
| Manual share | 20 % manual tokens | Higher/lower share | It is a course requirement, and manual material adds domains (literature, Wikisource) beyond news | Manual material is a small share, so overall domain mix is still web/news |

## Example: what cleaning does

Input (Hindi with HTML, a URL, an e-mail address and shouting):

```
<p>भारत   एक   सुंदर   देश   है।</p> देखें https://example.com/news अधिक जानकारी के लिए info@example.com पर लिखें!!!!!!!! यहाँ हर वाक्य में अलग शब्द हैं ताकि दोहराव जाँच पास हो जाए।
```

Output of `clean_text(..., lang="hindi", min_chars=20)`:

```
भारत एक सुंदर देश है। देखें अधिक जानकारी के लिए पर लिखें!!! यहाँ हर वाक्य में अलग शब्द हैं ताकि दोहराव जाँच पास हो जाए।
```

Rejected outright (return `None`): text shorter than 80 characters, mostly non-Devanagari text, and text whose unique-word ratio is below 0.65 (for example one phrase repeated 60 times).

Note the seam left by URL removal (`के लिए पर लिखें`): removing a token leaves the surrounding grammar broken. It is harmless at 500 M tokens but is a known cost of stripping instead of rewriting.

## Limitations and known issues

- Hindi and Nepali share a script, so the language filter cannot separate them. A small character-n-gram model trained from scratch on seed text for each language (allowed under the no-pretrained rule) would.
- The corpus-size figures in different reports describe different stages and are not reconciled in one place (see [known issues](known-issues.md)). A single machine-generated `corpus_statistics.json` should be the source of truth.
- The raw and processed corpora are not in git. They live in the project Google Drive folder referenced from the repository README; see [`data/README.md`](../data/README.md).
