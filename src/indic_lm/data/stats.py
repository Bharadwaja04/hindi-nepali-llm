#!/usr/bin/env python3
"""Compute corpus statistics for cleaned+split JSONL shards.

Reports (per language):
  - number of documents
  - total characters, total UTF-8 bytes
  - source breakdown (manual vs downloaded) in docs / chars / bytes
  - top source categories if present
  - approximate token count if a trained tokenizer JSON is supplied
  - average chars/token and bytes/token (when tokenizer given)
  - unknown-token rate on a held-out sample (when tokenizer given)

Writes a JSON stats file and prints a Markdown table to stdout.

Usage::

    python -m indic_lm.data.stats --lang nepali \
        --input /content/drive/MyDrive/LMA_Project/nepali/splits \
        --output /content/drive/MyDrive/LMA_Project/nepali/stats/nepali_stats.json \
        --tokenizer /content/drive/MyDrive/LMA_Project/nepali/tokenizer/nepali_tokenizer.json \
        [--heldout /content/drive/MyDrive/LMA_Project/nepali/splits/nepali_heldout.jsonl]
"""

import argparse
import json
import os
import sys
import time
from collections import Counter


def iter_jsonl(path):
    if os.path.isdir(path):
        for fn in sorted(os.listdir(path)):
            if fn.endswith(".jsonl"):
                with open(os.path.join(path, fn), "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            yield line
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield line


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    p.add_argument(
        "--input", required=True, help="Dir of jsonl shards (splits or cleaned)."
    )
    p.add_argument(
        "--output", required=True, help="Where to write the JSON stats file."
    )
    p.add_argument(
        "--tokenizer",
        default=None,
        help="Optional trained tokenizer JSON for token counts.",
    )
    p.add_argument(
        "--heldout", default=None, help="Held-out jsonl for unknown-token rate."
    )
    p.add_argument(
        "--sample-tokens",
        type=int,
        default=200_000,
        help="Max tokens to scan for unknown-token rate.",
    )
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    n_docs = 0
    n_chars = 0
    n_bytes = 0
    src_docs = Counter()
    src_chars = Counter()
    src_bytes = Counter()
    t0 = time.time()

    for line in iter_jsonl(args.input):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = rec.get("text", "")
        if not text:
            continue
        src = rec.get("source", "unknown")
        c = len(text)
        b = rec.get("n_bytes", len(text.encode("utf-8")))
        n_docs += 1
        n_chars += c
        n_bytes += b
        src_docs[src] += 1
        src_chars[src] += c
        src_bytes[src] += b
        if n_docs % 500_000 == 0:
            print(f"  scanned {n_docs:,} docs, {n_chars:,} chars ...", flush=True)

    stats = dict(
        lang=args.lang,
        n_docs=n_docs,
        n_chars=n_chars,
        n_bytes=n_bytes,
        n_mbytes=round(n_bytes / 1e6, 2),
        n_gbytes=round(n_bytes / 1e9, 4),
        source_breakdown={
            "docs": dict(src_docs),
            "chars": dict(src_chars),
            "bytes": dict(src_bytes),
        },
        secs=round(time.time() - t0, 2),
    )

    # token count with tokenizer
    if args.tokenizer and os.path.exists(args.tokenizer):
        from indic_lm.tokenizer import Tokenizer

        tok = Tokenizer.load(args.tokenizer)
        n_tokens = 0
        n_seen = 0
        for line in iter_jsonl(args.input):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("text", "")
            if not text:
                continue
            ids = tok.encode(text)
            n_tokens += len(ids)
            n_seen += 1
            if n_seen % 200_000 == 0:
                print(
                    f"  tokenized {n_seen:,} docs, {n_tokens:,} tokens ...", flush=True
                )
        stats["n_tokens"] = n_tokens
        stats["n_tokens_million"] = round(n_tokens / 1e6, 2)
        stats["avg_chars_per_token"] = round(n_chars / max(n_tokens, 1), 3)
        stats["avg_bytes_per_token"] = round(n_bytes / max(n_tokens, 1), 3)
        # source split in tokens
        src_tokens = Counter()
        for line in iter_jsonl(args.input):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("text", "")
            if not text:
                continue
            src = rec.get("source", "unknown")
            src_tokens[src] += len(tok.encode(text))
        stats["source_breakdown"]["tokens"] = dict(src_tokens)
        if n_tokens > 0:
            stats["manual_token_fraction"] = round(
                src_tokens.get("manual", 0) / n_tokens, 4
            )
            stats["downloaded_token_fraction"] = round(
                src_tokens.get("downloaded", 0) / n_tokens, 4
            )

        # unknown-token rate on held-out
        if args.heldout and os.path.exists(args.heldout):
            base = tok.config.base_vocab_size
            n_unk = 0
            n_tok = 0
            n_docs_h = 0
            for line in iter_jsonl(args.heldout):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ids = tok.encode(rec.get("text", ""))
                n_tok += len(ids)
                # unknowns = tokens that are raw single bytes < 256 which
                # never got merged (very rough proxy) -- a real unk rate
                # needs an <unk> token; we report fraction of singleton
                # byte tokens as "unmerged-byte fraction".
                n_unk += sum(1 for i in ids if i < base)
                n_docs_h += 1
                if n_tok >= args.sample_tokens:
                    break
            stats["heldout_unknown_fraction"] = round(n_unk / max(n_tok, 1), 4)
            stats["heldout_docs"] = n_docs_h
            stats["heldout_tokens_sampled"] = n_tok

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[stats] written -> {args.output}", flush=True)
    # Markdown summary
    print("\n| metric | value |")
    print("|---|---|")
    for k, v in stats.items():
        if k == "source_breakdown":
            print(f"| source_breakdown.docs | {v['docs']} |")
            print(f"| source_breakdown.chars | {v['chars']} |")
            if "tokens" in v:
                print(f"| source_breakdown.tokens | {v['tokens']} |")
            continue
        print(f"| {k} | {v} |")


if __name__ == "__main__":
    main()
