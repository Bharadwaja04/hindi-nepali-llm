import argparse
from pathlib import Path

from .tokenizer import Tokenizer

def read_text(path):
    path = Path(path)

    if path.is_file():
        return path.read_text(encoding="utf-8")

    return "\n".join(
        file.read_text(encoding="utf-8")
        for file in sorted(path.rglob("*.txt"))
    )

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained tokenizer."
    )

    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--text", required=True)

    args = parser.parse_args()

    tokenizer = Tokenizer.load(
        args.tokenizer
    )

    text = read_text(args.text)

    print(tokenizer.tokenization_stats(text))
    print(
        "round_trip:",
        tokenizer.verify_round_trip(text)
    )

if __name__ == "__main__":
    main()
