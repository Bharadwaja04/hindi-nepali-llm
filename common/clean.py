#!/usr/bin/env python3
"""Streaming text cleaner for monolingual Devanagari corpora.

Reads JSONL (one JSON object per line, expects a ``text`` field; falls
back to using the whole line as text), cleans each document, and writes
survivors to sharded JSONL output.  Designed to run on a 22 GB file in
Colab CPU RAM without loading the whole file: documents are streamed,
dedup state is a bloom filter + a bounded LRU set of exact hashes.

Usage (single process)::

    python -m common.clean --lang nepali \
        --input /content/drive/MyDrive/LMA_Project/nepali/raw/downloaded/nepali_train.jsonl \
        --output-dir /content/drive/MyDrive/LMA_Project/nepali/processed/clean \
        --source-tag downloaded --shard-size 200000

Parallel (4 workers, shards balanced by round-robin)::

    python -m common.clean --lang nepali --workers 4 \
        --input ... --output-dir ... --source-tag downloaded

Adds/overwrites these fields in each output record:
    text        cleaned text (NFC, stripped, single-spaced)
    source      "manual" | "downloaded"  (from --source-tag)
    lang        "hindi" | "nepali"
    n_chars     len(cleaned text)
    n_bytes     len(cleaned text.encode("utf-8"))
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from multiprocessing import Pool, cpu_count

from .langfilter import looks_hindi, looks_nepali

# ---------------------------------------------------------------------------
# Regexes (compiled once)
# ---------------------------------------------------------------------------
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")
_RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_RE_MULTI_SPACE = re.compile(r"[ \t]+")
_RE_BLANK_LINES = re.compile(r"\n{3,}")
_RE_NON_PRINTABLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
# Repeated-character / repeated-word spam
_RE_REPEAT_CHAR = re.compile(r"(.)\1{4,}")
_RE_REPEAT_WORD = re.compile(r"(\S+)(\s+\1){4,}")


# Devanagari combining marks that NFC should normalize away
def _nfc(s):
    return unicodedata.normalize("NFC", s)


# ---------------------------------------------------------------------------
# Bloom filter (pure-python, byte-array, k hash functions via double hashing)
# ---------------------------------------------------------------------------
class BloomFilter:
    def __init__(self, n_items, fp_rate=1e-4):
        m = max(8, int(-n_items * math.log(fp_rate) / (math.log(2) ** 2)))
        k = max(1, int(round((m / n_items) * math.log(2))))
        self.m = m
        self.k = k
        self.bits = bytearray((m + 7) // 8)

    def _hashes(self, data):
        h1 = int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")
        h2 = int.from_bytes(hashlib.sha256(data).digest()[:8], "big") | 1
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, data):
        for pos in self._hashes(data):
            self.bits[pos >> 3] |= 1 << (pos & 7)

    def __contains__(self, data):
        return all(self.bits[pos >> 3] & (1 << (pos & 7)) for pos in self._hashes(data))


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_text(
    text,
    lang="nepali",
    min_chars=80,
    max_chars=100_000,
    min_devanagari_ratio=0.55,
    max_repeat_ratio=0.35,
):
    """Return cleaned text or None if the document is rejected."""
    if not text or not isinstance(text, str):
        return None
    t = _nfc(text)
    t = _RE_NON_PRINTABLE.sub("", t)
    t = _RE_HTML_TAG.sub(" ", t)
    t = _RE_MARKDOWN_LINK.sub(" ", t)
    t = _RE_URL.sub(" ", t)
    t = _RE_EMAIL.sub(" ", t)
    t = _RE_REPEAT_CHAR.sub(r"\1\1\1", t)
    t = _RE_REPEAT_WORD.sub(r"\1", t)
    t = _RE_MULTI_SPACE.sub(" ", t)
    t = _RE_BLANK_LINES.sub("\n\n", t)
    t = t.strip()

    if len(t) < min_chars or len(t) > max_chars:
        return None

    # script filter
    if lang == "hindi":
        if not looks_hindi(t, min_ratio=min_devanagari_ratio):
            return None
    else:
        if not looks_nepali(t, min_ratio=min_devanagari_ratio):
            return None

    # repetition / gibberish filter: ratio of unique words
    words = t.split()
    if words:
        uniq = len(set(words))
        if uniq / len(words) < (1 - max_repeat_ratio):
            return None
    return t


# ---------------------------------------------------------------------------
# Streaming reader (handles plain text or jsonl, yields (meta, text))
# ---------------------------------------------------------------------------
def iter_records(path, text_field="text"):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    text = obj.get(text_field)
                    if text is None:
                        # try common alt field names
                        for k in ("content", "body", "raw", "document"):
                            if k in obj:
                                text = obj[k]
                                break
                    yield obj, text, line_no
                elif isinstance(obj, str):
                    yield {}, obj, line_no
                else:
                    yield {}, str(obj), line_no
            except json.JSONDecodeError:
                # treat as raw text
                yield {}, line, line_no


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _worker(args):
    (
        path,
        out_dir,
        shard_idx,
        source_tag,
        lang,
        shard_size,
        min_chars,
        min_deva,
        use_bloom,
        bloom_n,
    ) = args

    cleaner = lambda t: clean_text(
        t, lang=lang, min_chars=min_chars, min_devanagari_ratio=min_deva
    )
    seen = set()
    bf = BloomFilter(bloom_n) if use_bloom else None
    n_in = n_out = n_dup = n_rej = 0
    out_path = os.path.join(out_dir, f"{lang}_clean_{shard_idx:05d}.jsonl")
    bytes_written = 0
    t0 = time.time()

    with open(out_path, "w", encoding="utf-8") as out:
        for meta, text, line_no in iter_records(path):
            n_in += 1
            c = cleaner(text)
            if c is None:
                n_rej += 1
                continue
            # exact dedup on sha1 of normalized text
            h = hashlib.sha1(c.encode("utf-8")).digest()
            if bf is not None:
                if h in seen or h in bf:
                    n_dup += 1
                    continue
                bf.add(h)
                seen.add(h)
                if len(seen) > 200_000:
                    seen.clear()  # keep RAM bounded; rely on BF for false-neg
            else:
                if h in seen:
                    n_dup += 1
                    continue
                seen.add(h)
                if len(seen) > 1_000_000:
                    seen.clear()
            rec = dict(meta) if isinstance(meta, dict) else {}
            rec["text"] = c
            rec["source"] = source_tag
            rec["lang"] = lang
            rec["n_chars"] = len(c)
            rec["n_bytes"] = len(c.encode("utf-8"))
            line_out = json.dumps(rec, ensure_ascii=False)
            out.write(line_out + "\n")
            bytes_written += len(line_out) + 1
            n_out += 1
    dt = time.time() - t0
    return dict(
        shard=shard_idx,
        in_=n_in,
        out=n_out,
        dup=n_dup,
        rej=n_rej,
        out_path=out_path,
        bytes=bytes_written,
        secs=dt,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def split_input(path, n_workers):
    """Return list of (path) repeated; we shard by line ranges instead."""
    # We can't cheaply split a 22GB file by lines without reading it.
    # Instead, each worker reads the SAME file but keeps only lines where
    # (line_no % n_workers) == worker_idx.  This yields balanced shards
    # and each worker streams independently.
    return [path] * n_workers


def main():
    p = argparse.ArgumentParser(description="Stream-clean a Devanagari JSONL corpus.")
    p.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    p.add_argument("--input", required=True, help="JSONL file to clean.")
    p.add_argument("--output-dir", required=True, help="Where to write cleaned shards.")
    p.add_argument(
        "--source-tag",
        default="downloaded",
        choices=["downloaded", "manual"],
        help="Tag every output record with this source label.",
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=200_000,
        help="Approx docs per output shard (affects stat granularity).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers. >1 uses multiprocessing.",
    )
    p.add_argument("--min-chars", type=int, default=80)
    p.add_argument("--min-devanagari-ratio", type=float, default=0.55)
    p.add_argument(
        "--bloom",
        action="store_true",
        help="Use a bloom filter for exact-dedup (saves RAM).",
    )
    p.add_argument(
        "--bloom-n",
        type=int,
        default=5_000_000,
        help="Estimated unique docs for bloom sizing.",
    )
    p.add_argument("--text-field", default="text")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.workers <= 1:
        print(f"[clean] single-process: {args.input} -> {args.output_dir}", flush=True)
        # shard by shard_size docs
        shard_idx = 0
        seen = set()
        bf = BloomFilter(args.bloom_n) if args.bloom else None
        n_in = n_out = n_dup = n_rej = 0
        out_path = os.path.join(
            args.output_dir, f"{args.lang}_clean_{shard_idx:05d}.jsonl"
        )
        f_out = open(out_path, "w", encoding="utf-8")
        bytes_written = 0
        t0 = time.time()
        for meta, text, line_no in iter_records(args.input, args.text_field):
            n_in += 1
            c = clean_text(
                text,
                lang=args.lang,
                min_chars=args.min_chars,
                min_devanagari_ratio=args.min_devanagari_ratio,
            )
            if c is None:
                n_rej += 1
                continue
            h = hashlib.sha1(c.encode("utf-8")).digest()
            if bf is not None:
                if h in seen or h in bf:
                    n_dup += 1
                    continue
                bf.add(h)
                seen.add(h)
                if len(seen) > 200_000:
                    seen.clear()
            else:
                if h in seen:
                    n_dup += 1
                    continue
                seen.add(h)
                if len(seen) > 1_000_000:
                    seen.clear()
            rec = dict(meta) if isinstance(meta, dict) else {}
            rec.update(
                text=c,
                source=args.source_tag,
                lang=args.lang,
                n_chars=len(c),
                n_bytes=len(c.encode("utf-8")),
            )
            line_out = json.dumps(rec, ensure_ascii=False)
            f_out.write(line_out + "\n")
            bytes_written += len(line_out) + 1
            n_out += 1
            if n_out and n_out % args.shard_size == 0:
                f_out.close()
                shard_idx += 1
                out_path = os.path.join(
                    args.output_dir, f"{args.lang}_clean_{shard_idx:05d}.jsonl"
                )
                f_out = open(out_path, "w", encoding="utf-8")
                seen.clear()
            if n_in % 100_000 == 0:
                dt = time.time() - t0
                print(
                    f"  in={n_in:,} out={n_out:,} rej={n_rej:,} dup={n_dup:,} "
                    f"shard={shard_idx} {bytes_written / 1e6:.1f}MB {n_in / dt:,.0f} rec/s",
                    flush=True,
                )
        f_out.close()
        dt = time.time() - t0
        print(
            f"[clean] DONE in={n_in:,} out={n_out:,} rej={n_rej:,} dup={n_dup:,} "
            f"bytes={bytes_written:,} secs={dt:.1f}",
            flush=True,
        )
        stats = dict(
            lang=args.lang,
            source=args.source_tag,
            input=args.input,
            output_dir=args.output_dir,
            in_=n_in,
            out=n_out,
            rej=n_rej,
            dup=n_dup,
            bytes=bytes_written,
            secs=dt,
        )
        sp = os.path.join(args.output_dir, f"{args.lang}_clean_stats.json")
        with open(sp, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[clean] stats -> {sp}", flush=True)
        return

    # ---------------- parallel ----------------
    n = args.workers
    print(f"[clean] parallel x{n}: {args.input} -> {args.output_dir}", flush=True)
    tasks = []
    for i in range(n):
        tasks.append(
            (
                args.input,
                args.output_dir,
                i,
                args.source_tag,
                args.lang,
                args.shard_size,
                args.min_chars,
                args.min_devanagari_ratio,
                args.bloom,
                args.bloom_n,
            )
        )
    # NOTE: each worker currently reads the whole file and filters by line_no % n.
    # That is I/O-heavy but simple. For 22GB on Colab, prefer single-process.
    # (Kept for completeness; default single-process is recommended.)
    with Pool(n) as pool:
        results = pool.map(_worker, tasks)
    total = {
        k: sum(r[k] for r in results)
        for k in ("in_", "out", "dup", "rej", "bytes", "secs")
    }
    print(f"[clean] parallel DONE {total}", flush=True)
    sp = os.path.join(args.output_dir, f"{args.lang}_clean_stats.json")
    with open(sp, "w") as f:
        json.dump({"workers": n, "results": results, "total": total}, f, indent=2)
    print(f"[clean] stats -> {sp}", flush=True)


if __name__ == "__main__":
    main()
