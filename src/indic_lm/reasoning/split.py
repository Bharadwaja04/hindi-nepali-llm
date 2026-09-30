import json
import random
from collections import Counter, defaultdict
from pathlib import Path


RANDOM_SEED = 42

DATASETS = {
    "hindi": {
        "input": Path(
            "data/reasoning/hindi/hindi_reasoning_deduplicated.jsonl"
        ),
        "output_dir": Path("data/reasoning/hindi/splits"),
    },
    "nepali": {
        "input": Path(
            "data/reasoning/nepali/nepali_reasoning_deduplicated.jsonl"
        ),
        "output_dir": Path("data/reasoning/nepali/splits"),
    },
}


SPLIT_RATIOS = {
    "train": 0.80,
    "validation": 0.10,
    "test": 0.10,
}


def load_jsonl(path):
    records = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    return records

def split_category_records(records):
    """
    Split each category separately so category proportions
    remain approximately equal across train/validation/test.
    """
    grouped = defaultdict(list)

    for record in records:
        grouped[record["category"]].append(record)

    split_records = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for category, category_records in grouped.items():
        random.shuffle(category_records)

        total = len(category_records)

        train_end = int(total * SPLIT_RATIOS["train"])

        validation_end = train_end + int(
            total * SPLIT_RATIOS["validation"]
        )

        split_records["train"].extend(
            category_records[:train_end]
        )

        split_records["validation"].extend(
            category_records[train_end:validation_end]
        )

        split_records["test"].extend(
            category_records[validation_end:]
        )

    # Shuffle the actual lists, not the dictionary keys.
    for split_name in split_records:
        random.shuffle(split_records[split_name])

    return split_records


def save_jsonl(records, path):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def print_split_summary(language, split_records):
    print(f"\nLanguage: {language}")

    for split_name, records in split_records.items():
        category_counts = Counter(
            record["category"] for record in records
        )

        print(f"\n{split_name}: {len(records)} records")
        print(f"Categories: {dict(category_counts)}")


def process_dataset(language, input_path, output_dir):
    if not input_path.exists():
        print(f"Missing input file: {input_path}")
        return False

    records = load_jsonl(input_path)

    random.seed(RANDOM_SEED)
    split_records = split_category_records(records)

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in split_records.items():
        output_path = output_dir / f"{split_name}.jsonl"
        save_jsonl(split_data, output_path)
        print(f"Saved: {output_path}")

    print_split_summary(language, split_records)

    total_split_records = sum(
        len(records) for records in split_records.values()
    )

    if total_split_records != len(records):
        print("ERROR: Split record counts do not match input.")
        return False

    return True


def main():
    all_passed = True

    for language, paths in DATASETS.items():
        result = process_dataset(
            language=language,
            input_path=paths["input"],
            output_dir=paths["output_dir"],
        )

        all_passed = all_passed and result

    print("\nFINAL RESULT:")
    print("PASS" if all_passed else "FAIL")


if __name__ == "__main__":
    main()
