"""Command-line BPE tokenizer training entry point.

Streams text from disk (does not load the whole corpus into one string
if multiple files are given).  Trains on a configurable character budget
subsample (default 80M chars) -- standard practice for BPE; you do not
need to train on the full 500M-token corpus.
"""

import argparse
from pathlib import Path

from .tokenizer import Tokenizer


def read_text(path, char_budget=None):
    """Read text from a file or directory of .txt / .jsonl files.

    If ``char_budget`` is set and the total text would exceed it, reading
    stops early (a streaming prefix read), keeping RAM bounded.
    """
    path = Path(path)

    if path.is_file():
        if path.suffix == ".jsonl":
            return _read_jsonl_text(path, char_budget)
        with path.open("r", encoding="utf-8", errors="replace") as f:
            if char_budget is None:
                return f.read()
            chunks = []
            total = 0
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total >= char_budget:
                    break
            return "".join(chunks)

    files = sorted(list(path.rglob("*.txt")) + list(path.rglob("*.jsonl")))
    texts = []
    total = 0
    for file in files:
        remaining = (char_budget - total) if char_budget else None
        piece = read_text(file, remaining)
        texts.append(piece)
        total += len(piece)
        if char_budget and total >= char_budget:
            break
    return "\n".join(texts)


def _read_jsonl_text(path, char_budget=None):
    import json as _json

    texts = []
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
                t = obj.get("text") if isinstance(obj, dict) else str(obj)
            except _json.JSONDecodeError:
                t = line
            if t:
                texts.append(t)
                total += len(t)
                if char_budget and total >= char_budget:
                    break
    return "\n".join(texts)


def main():
    parser = argparse.ArgumentParser(
        description="Train a language-specific BPE tokenizer."
    )

    parser.add_argument(
        "--language",
        required=True,
        choices=["hindi", "nepali"],
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Text file or directory containing .txt/.jsonl files.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the tokenizer JSON.",
    )
    parser.add_argument(
        "--char-budget",
        type=int,
        default=80_000_000,
        help="Max characters to train BPE on (subsample). Default 80M.",
    )

    args = parser.parse_args()

    text = read_text(args.data, char_budget=args.char_budget)

    print(f"Language: {args.language}")
    print(f"Read characters (before subsample): {len(text):,}")
    print(f"Training vocabulary: {args.vocab_size:,}")
    print(f"Char budget for merges: {args.char_budget:,}")

    tokenizer = Tokenizer(
        language=args.language,
        vocab_size=args.vocab_size,
    )

    tokenizer.train(
        text,
        verbose=True,
        char_budget=args.char_budget,
    )

    tokenizer.save(args.output)
    print(f"Saved tokenizer to: {args.output}")


if __name__ == "__main__":
    main()
