import json
import unicodedata
from collections import Counter
from pathlib import Path


DATASETS = {
    "hindi": {
        "input": Path(
            "data/reasoning/hindi/hindi_reasoning_generated.jsonl"
        ),
        "output": Path(
            "data/reasoning/hindi/hindi_reasoning_deduplicated.jsonl"
        ),
    },
    "nepali": {
        "input": Path(
            "data/reasoning/nepali/nepali_reasoning_generated.jsonl"
        ),
        "output": Path(
            "data/reasoning/nepali/nepali_reasoning_deduplicated.jsonl"
        ),
    },
}


def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split()).strip()


def deduplicate_dataset(language, input_path, output_path):
    seen_texts = set()
    unique_records = []
    duplicate_count = 0

    with input_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            row = json.loads(line)

            normalized_text = normalize_text(row["text"])

            if normalized_text in seen_texts:
                duplicate_count += 1
                continue

            seen_texts.add(normalized_text)
            unique_records.append(row)

    with output_path.open("w", encoding="utf-8") as file:
        for row in unique_records:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    category_counts = Counter(
        row["category"] for row in unique_records
    )

    print(f"\nLanguage: {language}")
    print(f"Original records: {len(unique_records) + duplicate_count}")
    print(f"Unique records: {len(unique_records)}")
    print(f"Removed duplicates: {duplicate_count}")
    print(f"Categories: {dict(category_counts)}")
    print(f"Output: {output_path}")


def main():
    for language, paths in DATASETS.items():
        if not paths["input"].exists():
            print(f"Missing file: {paths['input']}")
            continue

        deduplicate_dataset(
            language=language,
            input_path=paths["input"],
            output_path=paths["output"],
        )


if __name__ == "__main__":
    main()
