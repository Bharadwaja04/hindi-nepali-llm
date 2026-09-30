#!/usr/bin/env python3
"""MinHash near-duplicate removal across cleaned shards.

Exact-dedup already happens in ``clean.py`` (sha1).  This module removes
*near* duplicates (same content, small edits / boilerplate replays) using
MinHash + LSH.  Runs after cleaning.

Usage::

    python -m indic_lm.data.dedup --lang nepali \
        --input /content/drive/MyDrive/LMA_Project/nepali/processed/clean \
        --output-dir /content/drive/MyDrive/LMA_Project/nepali/processed/dedup \
        --num-perm 128 --threshold 0.9 --shingles 5
"""

import argparse
import hashlib
import json
import os
import re
import time
from collections import defaultdict

WORD_RE = re.compile(r"\S+")


def shingle(text, k=5):
    words = WORD_RE.findall(text)
    if len(words) < k:
        return tuple(
            " ".join(words),
        )
    return tuple(" ".join(words[i : i + k]) for i in range(len(words) - k + 1))


def _h64(s):
    return int.from_bytes(
        hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big"
    )


def minhash(shingles, num_perm, seed=1):
    """Return a list of `num_perm` min-hash values (large ints)."""
    mh = []
    for i in range(num_perm):
        a = (1664525 * (i + 1) + seed) & 0xFFFFFFFFFFFFFFFF
        b = (1013904223 * (i + 7) + seed * 7) & 0xFFFFFFFFFFFFFFFF
        m = None
        for s in shingles:
            h = (a * _h64(s) + b) & 0xFFFFFFFFFFFFFFFF
            if m is None or h < m:
                m = h
        mh.append(m)
    return mh


class LSH:
    def __init__(self, num_perm, bands, threshold):
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self.threshold = threshold
        self.buckets = [defaultdict(list) for _ in range(bands)]

    def _band_keys(self, mh):
        keys = []
        for b in range(self.bands):
            start = b * self.rows
            chunk = mh[start : start + self.rows]
            keys.append((b, tuple(chunk)))
        return keys

    def insert(self, doc_id, mh):
        for key in self._band_keys(mh):
            self.buckets[key[0]][key[1]].append(doc_id)

    def query(self, mh):
        cand = set()
        for key in self._band_keys(mh):
            for d in self.buckets[key[0]].get(key[1], ()):
                cand.add(d)
        return cand


def jaccard(mh1, mh2):
    return sum(a == b for a, b in zip(mh1, mh2)) / len(mh1)


def iter_clean_files(input_dir):
    for fn in sorted(os.listdir(input_dir)):
        if fn.endswith(".jsonl") and "clean" in fn:
            yield os.path.join(input_dir, fn)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    p.add_argument("--input", required=True, help="Directory of cleaned shards.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-perm", type=int, default=128)
    p.add_argument("--bands", type=int, default=16)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--shingles", type=int, default=5)
    p.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Skip near-dedup for docs shorter than this (exact dedup already handled them).",
    )
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    lsh = LSH(args.num_perm, args.bands, args.threshold)
    sigs = {}  # doc_id -> minhash
    kept = 0
    removed = 0
    t0 = time.time()

    out_path = os.path.join(args.output_dir, f"{args.lang}_dedup.jsonl")
    out = open(out_path, "w", encoding="utf-8")
    doc_id = 0

    for fp in iter_clean_files(args.input):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = rec.get("text", "")
                if len(text) < args.min_chars:
                    out.write(line)
                    kept += 1
                    continue
                sh = shingle(text, args.shingles)
                mh = minhash(sh, args.num_perm)
                cands = lsh.query(mh)
                is_dup = False
                for c in cands:
                    if jaccard(sigs[c], mh) >= args.threshold:
                        is_dup = True
                        break
                if is_dup:
                    removed += 1
                else:
                    lsh.insert(doc_id, mh)
                    sigs[doc_id] = mh
                    doc_id += 1
                    out.write(line)
                    kept += 1
                if (kept + removed) % 20000 == 0:
                    dt = time.time() - t0
                    print(
                        f"  seen={kept + removed:,} kept={kept:,} near_dup={removed:,} "
                        f"{(kept + removed) / dt:,.0f} doc/s",
                        flush=True,
                    )
    out.close()
    dt = time.time() - t0
    print(f"[dedup] DONE kept={kept:,} removed={removed:,} secs={dt:.1f}", flush=True)
    with open(os.path.join(args.output_dir, f"{args.lang}_dedup_stats.json"), "w") as f:
        json.dump(
            dict(
                lang=args.lang,
                kept=kept,
                removed=removed,
                num_perm=args.num_perm,
                bands=args.bands,
                threshold=args.threshold,
                secs=dt,
            ),
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
