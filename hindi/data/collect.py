"""
Dataset collection manifest.

This script records the sources used for the language corpus.
It does not silently download arbitrary data.

Each source should have:
- name
- type
- URL/path
- manual/non-manual
- collection date
- notes
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path


SOURCES = [
    {
        "name": "TODO",
        "type": "public_corpus",
        "location": "TODO",
        "source_class": "non_manual",
        "collected_on": str(date.today()),
        "notes": "TODO",
    },
]


def save_manifest(
    output: Path
) -> None:

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            SOURCES,
            f,
            ensure_ascii=False,
            indent=2
        )


if __name__ == "__main__":

    save_manifest(
        Path(
            "data/source_manifest.json"
        )
    )