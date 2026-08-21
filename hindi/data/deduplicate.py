"""
Exact document-level deduplication.

The normalized text is hashed using SHA-256.
Only the first occurrence is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def text_hash(text: str) -> str:
    """
    Return a deterministic SHA-256 hash of normalized text.
    """

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def deduplicate_file(
    input_file: Path,
    output_file: Path,
) -> dict:

    seen = set()

    stats = {
        "input_documents": 0,
        "output_documents": 0,
        "duplicate_documents": 0,
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
                continue

            text = record.get(
                "text",
                ""
            )

            key = text_hash(text)

            if key in seen:

                stats[
                    "duplicate_documents"
                ] += 1

                continue

            seen.add(key)

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

    args = parser.parse_args()

    stats = deduplicate_file(
        Path(args.input),
        Path(args.output)
    )

    print(
        json.dumps(
            stats,
            indent=2
        )
    )


if __name__ == "__main__":
    main()