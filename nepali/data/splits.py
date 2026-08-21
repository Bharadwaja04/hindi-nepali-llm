#!/usr/bin/env python3
"""Create document-level train/validation/test splits from cleaned+deduped shards.

Splits are at the *document* level (not sentence/token level) to avoid
leakage.  Reads all cleaned (or deduped) JSONL shards in a directory,
shuffles deterministically (seeded), and writes three shards.

Usage::

    python -m common.splits --lang nepali \
        --input /content/drive/MyDrive/LMA_Project/nepali/processed/dedup \
        --output-dir /content/drive/MyDrive/LMA_Project/nepali/splits \
        --train 0.98 --val 0.01 --test 0.01 --seed 17
"""

import argparse
import json
import os
import random
import time


def iter_docs(input_dir):
    for fn in sorted(os.listdir(input_dir)):
        if not fn.endswith(".jsonl"):
            continue
        if (
            fn.startswith("nepali_dedup")
            or fn.startswith("hindi_dedup")
            or "clean" in fn
        ):
            with open(os.path.join(input_dir, fn), "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield line


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    p.add_argument(
        "--input", required=True, help="Dir of cleaned/deduped jsonl shards."
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train", type=float, default=0.98)
    p.add_argument("--val", type=float, default=0.01)
    p.add_argument("--test", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--buffer-size",
        type=int,
        default=2_000_000,
        help="Reservoir size for streamed shuffle (RAM-bound).",
    )
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # streamed reservoir shuffle (deterministic with seed)
    rng = random.Random(args.seed)
    reservoir = []
    n = 0
    t0 = time.time()
    for line in iter_docs(args.input):
        n += 1
        if len(reservoir) < args.buffer_size:
            reservoir.append(line)
        else:
            j = rng.randint(0, n - 1)
            if j < args.buffer_size:
                reservoir[j] = line
        if n % 500_000 == 0:
            print(f"  read {n:,} docs ...", flush=True)
    print(f"  total read {n:,}, reservoir {len(reservoir):,}", flush=True)
    rng.shuffle(reservoir)

    n_total = len(reservoir)
    n_train = int(n_total * args.train)
    n_val = int(n_total * args.val)
    # remainder -> test
    train = reservoir[:n_train]
    val = reservoir[n_train : n_train + n_val]
    test = reservoir[n_train + n_val :]

    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(args.output_dir, f"{args.lang}_{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for line in data:
                f.write(line)
        # also a small sample for tokenizer examples / held-out
        if name == "val":
            with open(
                os.path.join(args.output_dir, f"{args.lang}_heldout.jsonl"),
                "w",
                encoding="utf-8",
            ) as f:
                for line in data[:500]:
                    f.write(line)

    stats = dict(
        lang=args.lang,
        total=n_total,
        train=len(train),
        val=len(val),
        test=len(test),
        seed=args.seed,
        secs=time.time() - t0,
    )
    with open(
        os.path.join(args.output_dir, f"{args.lang}_splits_stats.json"), "w"
    ) as f:
        json.dump(stats, f, indent=2)
    print(f"[splits] {stats}", flush=True)


if __name__ == "__main__":
    main()
