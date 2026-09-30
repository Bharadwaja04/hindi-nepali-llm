import json
import unicodedata
from collections import Counter
from pathlib import Path


DATASETS = {
    "hindi": Path(
        "data/reasoning/hindi/hindi_reasoning_generated.jsonl"
    ),
    "nepali": Path(
        "data/reasoning/nepali/nepali_reasoning_generated.jsonl"
    ),
}


def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split()).strip()


def check_duplicates(language, path):
    records = []
    seen = {}
    duplicate_indices = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            row = json.loads(line)

            # Use the complete training text for duplicate detection.
            key = normalize_text(row["text"])

            if key in seen:
                duplicate_indices.append({
                    "line": line_number,
                    "duplicate_of_line": seen[key],
                    "id": row["id"],
                    "category": row["category"],
                })
            else:
                seen[key] = line_number

            records.append(row)

    category_counts = Counter(
        row["category"] for row in records
    )

    print(f"\nLanguage: {language}")
    print(f"File: {path}")
    print(f"Total records: {len(records)}")
    print(f"Unique texts: {len(seen)}")
    print(f"Duplicate texts: {len(duplicate_indices)}")
    print(f"Categories: {dict(category_counts)}")

    if duplicate_indices:
        print("\nFirst duplicate examples:")

        for duplicate in duplicate_indices[:10]:
            print(duplicate)


def main():
    for language, path in DATASETS.items():
        if not path.exists():
            print(f"Missing file: {path}")
            continue

        check_duplicates(language, path)


if __name__ == "__main__":
    main()
