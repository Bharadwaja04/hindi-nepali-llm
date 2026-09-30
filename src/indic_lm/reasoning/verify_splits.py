import json
from collections import Counter
from pathlib import Path


DATASETS = {
    "hindi": Path("data/reasoning/hindi/splits"),
    "nepali": Path("data/reasoning/nepali/splits"),
}

SPLITS = ["train", "validation", "test"]


def load_jsonl(path):
    records = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    return records


def verify_language(language, split_dir):
    all_ids = {}
    all_texts = {}
    split_records = {}

    print(f"\nLanguage: {language}")

    for split_name in SPLITS:
        path = split_dir / f"{split_name}.jsonl"

        if not path.exists():
            print(f"Missing file: {path}")
            return False

        records = load_jsonl(path)
        split_records[split_name] = records

        category_counts = Counter(
            record["category"] for record in records
        )

        print(f"\n{split_name}: {len(records)} records")
        print(f"Categories: {dict(category_counts)}")

        for record in records:
            record_id = str(record["id"])
            text = " ".join(record["text"].split())

            if record_id in all_ids:
                print(
                    f"Duplicate ID across splits: {record_id}"
                )
                return False

            if text in all_texts:
                print(
                    "Duplicate text across splits:"
                )
                print(f"  Current split: {split_name}")
                print(f"  Previous split: {all_texts[text]}")
                return False

            all_ids[record_id] = split_name
            all_texts[text] = split_name

    total_records = sum(
        len(records) for records in split_records.values()
    )

    print(f"\nTotal records across splits: {total_records}")
    print(f"Unique IDs: {len(all_ids)}")
    print(f"Unique texts: {len(all_texts)}")

    if total_records != len(all_ids):
        print("ERROR: Duplicate IDs detected.")
        return False

    if total_records != len(all_texts):
        print("ERROR: Duplicate texts detected.")
        return False

    print("Split verification: PASS")
    return True


def main():
    all_passed = True

    for language, split_dir in DATASETS.items():
        result = verify_language(language, split_dir)
        all_passed = all_passed and result

    print("\nFINAL RESULT:")
    print("PASS" if all_passed else "FAIL")


if __name__ == "__main__":
    main()
