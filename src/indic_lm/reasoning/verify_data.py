import json
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

EXPECTED_COUNTS = {
    "direct_comparison": 2000,
    "ordering": 2000,
    "transitive_comparison": 2500,
    "equality": 1500,
    "multi_hop": 1500,
    "basic_logic": 500,
}


def is_valid_field(value, field):
    """
    IDs may be strings or integers.
    All other fields must be non-empty strings.
    """
    if field == "id":
        return value is not None and str(value).strip() != ""

    return isinstance(value, str) and bool(value.strip())


def verify_record(row, language):
    required_fields = [
        "id",
        "language",
        "category",
        "question",
        "reasoning",
        "answer",
        "text",
    ]

    for field in required_fields:
        if field not in row:
            return False, f"missing field: {field}"

        if not is_valid_field(row[field], field):
            return False, f"empty field: {field}"

    if row["language"] != language:
        return False, "incorrect language label"

    expected_prefixes = {
        "hindi": ("प्रश्न:", "समाधान:", "उत्तर:"),
        "nepali": ("प्रश्न:", "समाधान:", "उत्तर:"),
    }

    question_prefix, reasoning_prefix, answer_prefix = (
        expected_prefixes[language]
    )

    text = row["text"]

    if not text.startswith(question_prefix):
        return False, "missing question section"

    if reasoning_prefix not in text:
        return False, "missing reasoning section"

    if answer_prefix not in text:
        return False, "missing answer section"

    if row["category"] not in EXPECTED_COUNTS:
        return False, "unknown category"

    return True, ""


def verify_dataset(language, path):
    total = 0
    passed = 0
    failures = []
    categories = Counter()
    ids = set()
    duplicate_ids = set()

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            total += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                failures.append({
                    "line": line_number,
                    "reason": f"invalid JSON: {error}",
                })
                continue

            record_id = row.get("id")

            # Check duplicate IDs.
            # Convert IDs to strings so 171 and "171" are treated consistently.
            normalized_id = str(record_id).strip()

            if normalized_id in ids:
                duplicate_ids.add(normalized_id)

            ids.add(normalized_id)

            category = row.get("category")
            categories[category] += 1

            ok, reason = verify_record(row, language)

            if ok:
                passed += 1
            else:
                failures.append({
                    "line": line_number,
                    "id": record_id,
                    "category": category,
                    "reason": reason,
                })

    print(f"\nLanguage: {language}")
    print(f"File: {path}")
    print(f"Total records: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(failures)}")
    print(f"Duplicate IDs: {len(duplicate_ids)}")
    print(f"Categories: {dict(categories)}")

    print("\nCategory checks:")

    category_counts_correct = True

    for category, expected in EXPECTED_COUNTS.items():
        actual = categories[category]
        status = "OK" if actual == expected else "WRONG"

        print(
            f"{category}: {actual} / {expected} [{status}]"
        )

        if actual != expected:
            category_counts_correct = False

    # Detect unexpected categories.
    unexpected_categories = set(categories) - set(EXPECTED_COUNTS)

    if unexpected_categories:
        category_counts_correct = False
        print(
            f"Unexpected categories: {sorted(unexpected_categories)}"
        )

    if failures:
        print("\nFirst failures:")

        for failure in failures[:10]:
            print(failure)

    if duplicate_ids:
        print("\nFirst duplicate IDs:")

        for duplicate_id in list(duplicate_ids)[:10]:
            print(duplicate_id)

    return (
        len(failures) == 0
        and len(duplicate_ids) == 0
        and category_counts_correct
        and total == 10000
    )


def main():
    all_passed = True

    for language, path in DATASETS.items():
        if not path.exists():
            print(f"Missing file: {path}")
            all_passed = False
            continue

        result = verify_dataset(language, path)
        all_passed = all_passed and result

    print("\nFINAL RESULT:")
    print("PASS" if all_passed else "FAIL")


if __name__ == "__main__":
    main()