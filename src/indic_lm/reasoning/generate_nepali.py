import json
import random
from pathlib import Path

SEED = 42
random.seed(SEED)

NUM_EXAMPLES = 10_000

OUTPUT_PATH = Path(
    "data/reasoning/nepali/nepali_reasoning_generated.jsonl"
)

NAMES = [
    "राम", "श्याम", "हरि", "गोपाल", "सीता",
    "गीता", "रीता", "सुनीता", "अनिल", "बिनोद",
    "सरिता", "माया", "कृष्ण", "प्रकाश", "मनोज",
    "सुरेश", "कमला", "निर्मला", "अमित", "विवेक",
]

RELATIONS = [
    ("उमेर", "वर्ष"),
    ("उचाइ", "सेन्टिमिटर"),
    ("मूल्य", "रुपैयाँ"),
    ("तौल", "किलोग्राम"),
    ("अंक", "अंक"),
]


def make_record(record_id, category, question, reasoning, answer):
    text = (
        f"प्रश्न: {question}\n\n"
        f"समाधान: {reasoning}\n\n"
        f"उत्तर: {answer}"
    )

    return {
        "id": record_id,
        "language": "nepali",
        "category": category,
        "question": question,
        "reasoning": reasoning,
        "answer": answer,
        "text": text,
    }


def direct_comparison(record_id):
    name_a, name_b = random.sample(NAMES, 2)
    relation, unit = random.choice(RELATIONS)

    value_a = random.randint(10, 90)
    value_b = random.randint(10, 90)

    if value_a == value_b:
        answer = "दुवै बराबर छन्।"
        reasoning = (
            f"{name_a} को {relation} {value_a} {unit} र "
            f"{name_b} को {relation} पनि {value_b} {unit} छ। "
            "त्यसैले दुवै बराबर छन्।"
        )
    elif value_a > value_b:
        answer = f"{name_a} को {relation} बढी छ।"
        reasoning = (
            f"{name_a} को {relation} {value_a} {unit} छ भने "
            f"{name_b} को {relation} {value_b} {unit} छ। "
            f"{value_a} ठूलो भएकाले {name_a} को {relation} बढी छ।"
        )
    else:
        answer = f"{name_b} को {relation} बढी छ।"
        reasoning = (
            f"{name_a} को {relation} {value_a} {unit} छ भने "
            f"{name_b} को {relation} {value_b} {unit} छ। "
            f"{value_b} ठूलो भएकाले {name_b} को {relation} बढी छ।"
        )

    question = (
        f"{name_a} को {relation} {value_a} {unit} र "
        f"{name_b} को {relation} {value_b} {unit} छ। "
        f"कसको {relation} बढी छ?"
    )

    return make_record(
        record_id,
        "direct_comparison",
        question,
        reasoning,
        answer,
    )


def ordering(record_id):
    names = random.sample(NAMES, 3)
    relation, unit = random.choice(RELATIONS)

    values = sorted(random.sample(range(10, 90), 3), reverse=True)

    data = dict(zip(names, values))

    question = (
        f"{names[0]} को {relation} {data[names[0]]} {unit}, "
        f"{names[1]} को {relation} {data[names[1]]} {unit}, "
        f"र {names[2]} को {relation} {data[names[2]]} {unit} छ। "
        f"सबैलाई {relation} को आधारमा ठूलोदेखि सानो क्रममा राख्नुहोस्।"
    )

    answer = " > ".join(names)

    reasoning = (
        f"{data[names[0]]} > {data[names[1]]} > "
        f"{data[names[2]]} भएकाले क्रम "
        f"{names[0]}, {names[1]}, {names[2]} हुन्छ।"
    )

    return make_record(
        record_id,
        "ordering",
        question,
        reasoning,
        answer,
    )


def transitive_comparison(record_id):
    names = random.sample(NAMES, 3)
    relation, unit = random.choice(RELATIONS)

    value_a = random.randint(40, 80)
    value_b = random.randint(20, value_a - 1)
    value_c = random.randint(10, value_b - 1)

    question = (
        f"{names[0]} को {relation} {names[1]} भन्दा बढी छ। "
        f"{names[1]} को {relation} {names[2]} भन्दा बढी छ। "
        f"त्यसैले कसको {relation} सबैभन्दा बढी छ?"
    )

    reasoning = (
        f"{names[0]} > {names[1]} र "
        f"{names[1]} > {names[2]} भएकाले "
        f"{names[0]} > {names[2]} हुन्छ।"
    )

    answer = f"{names[0]} को {relation} सबैभन्दा बढी छ।"

    return make_record(
        record_id,
        "transitive_comparison",
        question,
        reasoning,
        answer,
    )


def equality(record_id):
    name_a, name_b = random.sample(NAMES, 2)
    relation, unit = random.choice(RELATIONS)
    value = random.randint(10, 90)

    question = (
        f"{name_a} र {name_b} दुवैको {relation} "
        f"{value} {unit} छ। के दुवैको {relation} बराबर छ?"
    )

    reasoning = (
        f"दुवैको {relation} {value} {unit} भएकाले "
        "उनीहरूको मान समान छ।"
    )

    answer = "हो, दुवैको मान बराबर छ।"

    return make_record(
        record_id,
        "equality",
        question,
        reasoning,
        answer,
    )


def multi_hop(record_id):
    names = random.sample(NAMES, 3)
    relation, unit = random.choice(RELATIONS)

    first = random.randint(20, 50)
    second = first + random.randint(5, 15)
    third = second + random.randint(5, 15)

    question = (
        f"{names[0]} को {relation} {first} {unit} छ। "
        f"{names[1]} को {relation} {names[0]} भन्दा बढी छ। "
        f"{names[2]} को {relation} {names[1]} भन्दा बढी छ। "
        f"सबैभन्दा बढी {relation} कसको छ?"
    )

    reasoning = (
        f"{names[1]} को {relation} {names[0]} भन्दा बढी छ र "
        f"{names[2]} को {relation} {names[1]} भन्दा बढी छ। "
        f"त्यसैले {names[2]} को {relation} सबैभन्दा बढी छ।"
    )

    answer = f"{names[2]} को {relation} सबैभन्दा बढी छ।"

    return make_record(
        record_id,
        "multi_hop",
        question,
        reasoning,
        answer,
    )


def basic_logic(record_id):
    name_a, name_b = random.sample(NAMES, 2)

    question = (
        f"यदि {name_a} को अंक {name_b} को अंकभन्दा कम छ भने, "
        f"कसको अंक बढी हुन्छ?"
    )

    reasoning = (
        f"{name_a} को अंक {name_b} को अंकभन्दा कम भएकाले "
        f"{name_b} को अंक बढी हुन्छ।"
    )

    answer = f"{name_b} को अंक बढी हुन्छ।"

    return make_record(
        record_id,
        "basic_logic",
        question,
        reasoning,
        answer,
    )


def generate_dataset():
    records = []
    record_id = 1

    generators = [
        (direct_comparison, 2000),
        (ordering, 2000),
        (transitive_comparison, 2500),
        (equality, 1500),
        (multi_hop, 1500),
        (basic_logic, 500),
    ]

    for generator, count in generators:
        for _ in range(count):
            records.append(generator(record_id))
            record_id += 1

    random.shuffle(records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Generated examples: {len(records)}")
    print(f"Output file: {OUTPUT_PATH}")
    print("Language: Nepali")
    print("Seed:", SEED)


if __name__ == "__main__":
    generate_dataset()
