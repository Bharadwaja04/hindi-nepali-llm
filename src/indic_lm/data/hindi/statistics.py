"""
Corpus statistics for Phase 1.

Computes:
- documents
- words
- characters
- UTF-8 bytes
- source/manual category
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def file_statistics(
    path: Path,
    source: str
) -> dict:

    documents = 0
    words = 0
    characters = 0
    utf8_bytes = 0

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                continue

            text = record.get(
                "text",
                ""
            )

            if not isinstance(
                text,
                str
            ):
                continue

            documents += 1
            words += len(
                text.split()
            )
            characters += len(text)
            utf8_bytes += len(
                text.encode("utf-8")
            )

    return {
        "file": str(path),
        "source": source,
        "documents": documents,
        "words": words,
        "characters": characters,
        "utf8_bytes": utf8_bytes,
    }


def corpus_statistics(
    files
) -> dict:

    result = {
        "documents": 0,
        "words": 0,
        "characters": 0,
        "utf8_bytes": 0,
        "manual_documents": 0,
        "non_manual_documents": 0,
        "files": [],
    }

    for path, source in files:

        stats = file_statistics(
            path,
            source
        )

        result["files"].append(
            stats
        )

        result["documents"] += (
            stats["documents"]
        )

        result["words"] += (
            stats["words"]
        )

        result["characters"] += (
            stats["characters"]
        )

        result["utf8_bytes"] += (
            stats["utf8_bytes"]
        )

        if source == "manual":

            result[
                "manual_documents"
            ] += stats["documents"]

        else:

            result[
                "non_manual_documents"
            ] += stats["documents"]

    return result