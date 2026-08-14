import argparse
from pathlib import Path

from .tokenizer import Tokenizer

def read_text(path):
    path = Path(path)

    if path.is_file():
        return path.read_text(encoding="utf-8")

    texts = []
    for file in sorted(path.rglob("*.txt")):
        texts.append(
            file.read_text(encoding="utf-8")
        )

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
        help="Text file or directory containing .txt files.",
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

    args = parser.parse_args()

    text = read_text(args.data)

    print(f"Language: {args.language}")
    print(f"Characters: {len(text):,}")
    print(f"Training vocabulary: {args.vocab_size:,}")

    tokenizer = Tokenizer(
        language=args.language,
        vocab_size=args.vocab_size,
    )

    tokenizer.train(
        text,
        verbose=True
    )

    tokenizer.save(args.output)

    print(f"Saved tokenizer to: {args.output}")
