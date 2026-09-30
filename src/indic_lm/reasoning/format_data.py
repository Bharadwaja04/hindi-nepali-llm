import json
from pathlib import Path


def convert_file(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with input_path.open("r", encoding="utf-8") as infile, \
            output_path.open("w", encoding="utf-8") as outfile:

        for line_number, line in enumerate(infile, start=1):
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            formatted_text = (
                f"प्रश्न: {record['question']}\n\n"
                f"समाधान: {record['reasoning']}\n\n"
                f"उत्तर: {record['answer']}"
            )

            output_record = {
                "id": record["id"],
                "language": record["language"],
                "category": record["category"],
                "text": formatted_text,
            }

            outfile.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False,
                ) + "\n"
            )

            count += 1

    print(f"Converted {count} records")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    convert_file(
        "data/reasoning/hindi/hindi_reasoning_raw.jsonl",
        "data/reasoning/hindi/hindi_reasoning_train.jsonl",
    )

    convert_file(
        "data/reasoning/nepali/nepali_reasoning_raw.jsonl",
        "data/reasoning/nepali/nepali_reasoning_train.jsonl",
    )
