"""Scalable byte-level BPE trainer.

Trains on a subsample of the corpus and works on *unique pre-tokens
weighted by frequency* (the minbpe/tiktoken approach), not on every
token occurrence.  This makes early merges of very common byte pairs
cheap: a pair that appears in a word repeated 100k times is processed
once, not 100k times.

Algorithm:
  1. Subsample text to ``char_budget`` chars.
  2. NFC-normalize + regex pre-tokenize into pieces.
  3. Encode each unique piece to bytes, count its frequency.
  4. Maintain pair frequencies = sum over unique words containing the
     pair of (word_freq * occurrences_of_pair_in_word).
  5. For each merge, find the best pair, rebuild only the unique words
     that contain it, and incrementally update pair counts.
  6. Use a max-heap with lazy deletion for best-pair selection.
"""

import heapq
from collections import Counter

from .vocab import build_vocab


class BPETrainer:
    """Train byte-level BPE from scratch on regex-pre-tokenized text."""

    def __init__(self, config, pretokenizer):
        self.config = config
        self.pretokenizer = pretokenizer
        self.merges = {}
        self.vocab = build_vocab(self.merges, config.base_vocab_size)

    def _read_subsample(self, text, char_budget):
        n = len(text)
        if n <= char_budget:
            return text
        start = (n - char_budget) // 2
        return text[start : start + char_budget]

    @staticmethod
    def _word_pairs(word):
        """Yield consecutive pairs in a word (list of ids)."""
        for i in range(len(word) - 1):
            yield (word[i], word[i + 1])

    def train(self, text, verbose=True, char_budget=80_000_000):
        self.config.validate()

        sub = self._read_subsample(text, char_budget)
        if verbose:
            print(
                f"[trainer] subsample: {len(sub):,} chars (of {len(text):,} total)",
                flush=True,
            )

        sub = self.pretokenizer.normalize(sub)
        pieces = self.pretokenizer.split(sub)

        # unique pre-tokens -> byte sequence + frequency
        word_freq = Counter()
        for p in pieces:
            if p:
                word_freq[bytes(p.encode("utf-8"))] += 1

        # represent each unique word as a list of byte ids
        words = [list(w) for w in word_freq.keys()]
        freqs = list(word_freq.values())
        n_unique = len(words)
        n_tokens = sum(len(w) * f for w, f in zip(words, freqs))
        if verbose:
            print(
                f"[trainer] {n_unique:,} unique pre-tokens, "
                f"{n_tokens:,} total bytes (weighted)",
                flush=True,
            )

        num_merges = self.config.vocab_size - self.config.base_vocab_size
        if num_merges <= 0:
            return self.merges, self.vocab

        # ---- pair counts (weighted) + pair -> set of word indices ----
        pair_counts = Counter()
        pair_to_words = {}

        for wi, w in enumerate(words):
            f = freqs[wi]
            seen_in_word = set()
            for pair in self._word_pairs(w):
                pair_counts[pair] += f
                seen_in_word.add(pair)
            for pair in seen_in_word:
                pair_to_words.setdefault(pair, set()).add(wi)

        # ---- max-heap of (-count, pair) ----
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        self.merges = {}

        for step in range(num_merges):
            # pop best non-stale entry
            best_pair = None
            best_count = -1
            while heap:
                neg_c, p = heapq.heappop(heap)
                c = -neg_c
                cur = pair_counts.get(p, 0)
                if c != cur:
                    continue  # stale
                best_pair = p
                best_count = c
                break

            if best_pair is None or best_count < 1:
                break

            new_id = self.config.base_vocab_size + step
            self.merges[best_pair] = new_id

            if verbose and (step < 10 or step % 500 == 0 or step == num_merges - 1):
                print(
                    f"[trainer] merge {step + 1:>6}/{num_merges} | "
                    f"pair={best_pair} | id={new_id} | freq={best_count}",
                    flush=True,
                )

            p0, p1 = best_pair
            affected = pair_to_words.get(best_pair, set())
            touched = set()

            for wi in list(affected):
                w = words[wi]
                f = freqs[wi]
                # rebuild word, merging best_pair
                new_w = []
                i = 0
                L = len(w)
                while i < L:
                    if i < L - 1 and w[i] == p0 and w[i + 1] == p1:
                        new_w.append(new_id)
                        i += 2
                    else:
                        new_w.append(w[i])
                        i += 1
                words[wi] = new_w

                # decrement old pairs (weighted by freq, once per pair per word)
                old_pairs_seen = set()
                for pair in self._word_pairs(w):
                    if pair not in old_pairs_seen:
                        old_pairs_seen.add(pair)
                        pair_counts[pair] -= f
                        touched.add(pair)
                        if pair_counts[pair] <= 0:
                            pair_counts.pop(pair, None)
                            pair_to_words.pop(pair, None)

                # increment new pairs
                new_pairs_seen = set()
                for pair in self._word_pairs(new_w):
                    if pair not in new_pairs_seen:
                        new_pairs_seen.add(pair)
                        pair_counts[pair] = pair_counts.get(pair, 0) + f
                        pair_to_words.setdefault(pair, set()).add(wi)
                        touched.add(pair)

            pair_counts.pop(best_pair, None)
            pair_to_words.pop(best_pair, None)
            touched.discard(best_pair)

            # push touched pairs with fresh counts
            for pair in touched:
                c = pair_counts.get(pair, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, pair))

        self.vocab = build_vocab(self.merges, self.config.base_vocab_size)
        if verbose:
            print(
                f"[trainer] done: {len(self.merges):,} merges, "
                f"vocab={self.config.vocab_size}",
                flush=True,
            )
        return self.merges, self.vocab
