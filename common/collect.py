#!/usr/bin/env python3
"""Download (80%) monolingual corpora from Hugging Face.

Pulls Hindi or Nepali text datasets via ``datasets`` (server-side streaming
mode, so nothing is downloaded to your laptop -- Colab pulls directly from
HF).  Writes JSONL shards tagged ``source: downloaded``.

Supported sources (configure via --sources, comma-separated):
  - mc4    : mC4 (C4 multilingual), split ``hi`` or ``ne``
  - oscar  : OSCAR 2301, split ``hi`` or ``ne``
  - wiki   : Wikipedia, split ``hi`` or ``ne``
  - indiccorp : AI4Bharat IndicCorp (if available on HF)

Usage::

    python -m common.collect --lang hindi --sources mc4,wiki \
        --output-dir /content/drive/MyDrive/LMA_Project/hindi/raw/downloaded \
        --max-chars 4_000_000_000 --shard-size 250_000
"""

import argparse
import json
import os
import time


def _stream_hf(
    dataset_name,
    config,
    split,
    lang,
    max_chars,
    shard_size,
    out_dir,
    source_tag,
    text_field="text",
):
    """Stream a HF dataset and write JSONL shards. Returns (n_docs, n_chars)."""
    from datasets import load_dataset

    shard_idx = 0
    n_docs = 0
    n_chars = 0
    shard_path = os.path.join(
        out_dir, f"{lang}_{source_tag}_{dataset_name}_{shard_idx:05d}.jsonl"
    )
    f_out = open(shard_path, "w", encoding="utf-8")

    try:
        if config:
            ds = load_dataset(dataset_name, config, split=split, streaming=True)
        else:
            ds = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:
        print(f"[collect] could not load {dataset_name}/{config}: {e}", flush=True)
        f_out.close()
        os.remove(shard_path)
        return 0, 0

    t0 = time.time()
    for rec in ds:
        text = rec.get(text_field)
        if not text or not isinstance(text, str):
            continue
        obj = {
            "text": text,
            "source": source_tag,
            "lang": lang,
            "dataset": dataset_name,
            "n_chars": len(text),
            "n_bytes": len(text.encode("utf-8")),
        }
        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        n_docs += 1
        n_chars += len(text)
        if n_docs % shard_size == 0:
            f_out.close()
            shard_idx += 1
            shard_path = os.path.join(
                out_dir, f"{lang}_{source_tag}_{dataset_name}_{shard_idx:05d}.jsonl"
            )
            f_out = open(shard_path, "w", encoding="utf-8")
        if n_docs % 100_000 == 0:
            dt = time.time() - t0
            print(
                f"  [{dataset_name}] {n_docs:,} docs, {n_chars:,} chars "
                f"({n_chars / 1e9:.2f}GB) {n_docs / dt:,.0f} rec/s",
                flush=True,
            )
        if max_chars and n_chars >= max_chars:
            break
    f_out.close()
    dt = time.time() - t0
    print(
        f"[collect] {dataset_name}: {n_docs:,} docs, {n_chars:,} chars "
        f"({n_chars / 1e9:.2f}GB) in {dt:.1f}s",
        flush=True,
    )
    return n_docs, n_chars


SOURCES = {
    "mc4": (
        "allenai/c4",
        "multilingual.c4",
        None,
    ),  # config hi/ne via split? actually mc4 uses lang config
    "wiki": ("wikimedia/wikipedia", "20231101.hi", None),  # config per lang
    "oscar": ("oscar-corpus/OSCAR-2301", "hi", None),
}

# Better: define per-lang source configs explicitly
LANG_SOURCES = {
    "hindi": {
        "mc4": ("allenai/c4", "multilingual.c4", "train"),  # filter lang=hi downstream
        "wiki": ("wikimedia/wikipedia", "20231101.hi", "train"),
        "oscar": ("oscar-corpus/OSCAR-2301", "hi", "train"),
    },
    "nepali": {
        "mc4": ("allenai/c4", "multilingual.c4", "train"),  # filter lang=ne downstream
        "wiki": ("wikimedia/wikipedia", "20231101.ne", "train"),
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    p.add_argument(
        "--sources",
        default="mc4,wiki",
        help="Comma-separated source names from: mc4,wiki,oscar",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--max-chars",
        type=int,
        default=4_000_000_000,
        help="Stop after this many total characters (default 4GB).",
    )
    p.add_argument("--shard-size", type=int, default=250_000)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    avail = LANG_SOURCES.get(args.lang, {})
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    total_docs = total_chars = 0
    budget = args.max_chars

    for s in sources:
        if s not in avail:
            print(
                f"[collect] skipping unknown source '{s}' for {args.lang}", flush=True
            )
            continue
        ds_name, config, split = avail[s]
        remaining = budget - total_chars if budget else None
        nd, nc = _stream_hf(
            ds_name,
            config,
            split,
            args.lang,
            remaining,
            args.shard_size,
            args.output_dir,
            source_tag="downloaded",
        )
        total_docs += nd
        total_chars += nc
        if budget and total_chars >= budget:
            print(f"[collect] reached char budget ({budget:,})", flush=True)
            break

    print(
        f"[collect] TOTAL: {total_docs:,} docs, {total_chars:,} chars "
        f"({total_chars / 1e9:.2f}GB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
