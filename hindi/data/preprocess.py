"""
Language-specific text preprocessing.

Responsibilities:
1. Read JSONL documents.
2. Validate text records.
3. Normalize Unicode.
4. Normalize whitespace.
5. Remove empty/invalid documents.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


def normalize_text(text: str) -> str:
    """
    Normalize Unicode and whitespace while preserving language content.

    NFC is used so canonically equivalent Unicode sequences have
    a consistent representation.
    """

    text = unicodedata.normalize(
        "NFC",
        text
    )

    # Normalize Unicode whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def valid_text(text: object) -> bool:
    """
    Return True if text is a usable non-empty string.
    """

    if not isinstance(text, str):
        return False

    text = text.strip()

    if not text:
        return False

    return True


def preprocess_file(
    input_file: Path,
    output_file: Path,
) -> dict:
    """
    Clean one JSONL file and write the normalized records.

    Returns processing statistics.
    """

    stats = {
        "input_documents": 0,
        "output_documents": 0,
        "invalid_documents": 0,
        "characters_before": 0,
        "characters_after": 0,
    }

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with (
        input_file.open(
            "r",
            encoding="utf-8"
        ) as src,
        output_file.open(
            "w",
            encoding="utf-8"
        ) as dst
    ):

        for line in src:

            stats[
                "input_documents"
            ] += 1

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                stats[
                    "invalid_documents"
                ] += 1
                continue

            text = record.get(
                "text",
                ""
            )

            if not valid_text(text):
                stats[
                    "invalid_documents"
                ] += 1
                continue

            stats[
                "characters_before"
            ] += len(text)

            text = normalize_text(text)

            if not text:
                stats[
                    "invalid_documents"
                ] += 1
                continue

            stats[
                "characters_after"
            ] += len(text)

            record["text"] = text

            dst.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

            stats[
                "output_documents"
            ] += 1

    return stats


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    args = parser.parse_args()

    stats = preprocess_file(
        Path(args.input),
        Path(args.output)
    )

    print(
        json.dumps(
            stats,
            indent=2,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()