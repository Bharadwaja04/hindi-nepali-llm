"""
Deterministic train/validation/test splitting.

Default:
    train = 98%
    validation = 1%
    test = 1%

A fixed seed makes the split reproducible.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def split_documents(
    input_file: Path,
    output_dir: Path,
    train_ratio: float = 0.98,
    val_ratio: float = 0.01,
    test_ratio: float = 0.01,
    seed: int = 42,
):

    assert abs(
        train_ratio +
        val_ratio +
        test_ratio - 1.0
    ) < 1e-8

    records = []

    with input_file.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            try:
                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError:
                continue

    rng = random.Random(seed)

    rng.shuffle(records)

    n = len(records)

    train_end = int(
        n * train_ratio
    )

    val_end = train_end + int(
        n * val_ratio
    )

    splits = {
        "train": records[:train_end],
        "validation": records[
            train_end:val_end
        ],
        "test": records[
            val_end:
        ],
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for name, items in splits.items():

        output_file = (
            output_dir /
            f"{name}.jsonl"
        )

        with output_file.open(
            "w",
            encoding="utf-8"
        ) as f:

            for record in items:

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

        print(
            name,
            len(items)
        )


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

    split_documents(
        Path(args.input),
        Path(args.output)
    )


if __name__ == "__main__":
    main()