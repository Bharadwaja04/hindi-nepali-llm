"""
Lightweight language/script filtering for Hindi and Nepali.

Hindi:
    Devanagari + common Indic punctuation/characters.

Nepali:
    Devanagari + common Indic punctuation/characters.

The filter is intentionally conservative and reports statistics
rather than silently modifying text.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEVANAGARI = re.compile(
    r"[\u0900-\u097F]"
)


def devanagari_ratio(
    text: str
) -> float:

    letters = [
        c for c in text
        if c.isalpha()
    ]

    if not letters:
        return 0.0

    devanagari = sum(
        bool(DEVANAGARI.match(c))
        for c in letters
    )

    return (
        devanagari /
        len(letters)
    )


def language_passes(
    text: str,
    threshold: float = 0.50
) -> bool:

    return (
        devanagari_ratio(text)
        >= threshold
    )


def filter_file(
    input_file: Path,
    output_file: Path,
    threshold: float = 0.50,
) -> dict:

    stats = {
        "input_documents": 0,
        "accepted_documents": 0,
        "rejected_documents": 0,
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

            record = json.loads(line)

            text = record.get(
                "text",
                ""
            )

            if language_passes(
                text,
                threshold
            ):

                dst.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                stats[
                    "accepted_documents"
                ] += 1

            else:

                stats[
                    "rejected_documents"
                ] += 1

    return stats


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50
    )

    args = parser.parse_args()

    stats = filter_file(
        Path(args.input),
        Path(args.output),
        args.threshold
    )

    print(
        json.dumps(
            stats,
            indent=2
        )
    )


if __name__ == "__main__":
    main()