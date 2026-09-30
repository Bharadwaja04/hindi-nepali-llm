import json
import random
from pathlib import Path


SEED = 42
NUM_EXAMPLES = 10_000

OUTPUT_PATH = Path(
    "data/reasoning/hindi/hindi_reasoning_generated.jsonl"
)

random.seed(SEED)


NAMES = [
    "राम", "सीता", "मोहन", "गीता", "अमित",
    "सीमा", "रवि", "कविता", "अनिल", "पूजा",
    "सुरेश", "नेहा", "विकास", "राधा", "अजय",
    "सोनम", "विनय", "मीना", "रोहित", "प्रिया",
]


RELATIONS = [
    ("उम्र", "वर्ष", "उम्र"),
    ("ऊँचाई", "सेंटीमीटर", "ऊँचाई"),
    ("कीमत", "रुपये", "कीमत"),
    ("वजन", "किलोग्राम", "वजन"),
    ("अंक", "अंक", "अंक"),
]


def make_record(record_id, category, question, reasoning, answer):
    return {
        "id": f"hi_{record_id:05d}",
        "language": "hindi",
        "category": category,
        "question": question,
        "reasoning": reasoning,
        "answer": answer,
        "text": (
            f"प्रश्न: {question}\n\n"
            f"समाधान: {reasoning}\n\n"
            f"उत्तर: {answer}"
        ),
    }


def direct_comparison(record_id):
    a, b = random.sample(NAMES, 2)
    relation, unit, relation_word = random.choice(RELATIONS)

    value_a = random.randint(5, 100)
    value_b = random.randint(5, 100)

    if value_a == value_b:
        answer = "दोनों समान हैं"
        reasoning = (
            f"{a} की {relation} {value_a} {unit} है और "
            f"{b} की {relation} भी {value_b} {unit} है। "
            "इसलिए दोनों समान हैं।"
        )
    elif value_a > value_b:
        answer = a
        reasoning = (
            f"{value_a} > {value_b}, इसलिए "
            f"{a} की {relation_word} अधिक है।"
        )
    else:
        answer = b
        reasoning = (
            f"{value_b} > {value_a}, इसलिए "
            f"{b} की {relation_word} अधिक है।"
        )

    question = (
        f"{a} की {relation} {value_a} {unit} है और "
        f"{b} की {relation} {value_b} {unit} है। "
        f"किसकी {relation_word} अधिक है या क्या दोनों समान हैं?"
    )

    return make_record(
        record_id,
        "direct_comparison",
        question,
        reasoning,
        answer,
    )


def ordering(record_id):
    a, b, c = random.sample(NAMES, 3)
    relation, unit, relation_word = random.choice(RELATIONS)

    values = {
        a: random.randint(5, 100),
        b: random.randint(5, 100),
        c: random.randint(5, 100),
    }

    ordered = sorted(values, key=values.get, reverse=True)
    question_order = random.choice(["सबसे अधिक", "सबसे कम"])

    if question_order == "सबसे अधिक":
        answer = ordered[0]
        comparison = " > ".join(ordered)
    else:
        answer = ordered[-1]
        comparison = " < ".join(ordered)

    question = (
        f"{a}, {b} और {c} की {relation} क्रमशः "
        f"{values[a]}, {values[b]} और {values[c]} {unit} है। "
        f"इनमें {question_order} {relation_word} किसकी है?"
    )

    reasoning = (
        f"मानों का क्रम {comparison} है। "
        f"इसलिए उत्तर {answer} है।"
    )

    return make_record(
        record_id,
        "ordering",
        question,
        reasoning,
        answer,
    )


def transitive_comparison(record_id):
    a, b, c = random.sample(NAMES, 3)

    value_a = random.randint(20, 80)
    value_b = random.randint(10, value_a - 1)
    value_c = random.randint(1, value_b - 1)

    question_type = random.choice(["सबसे बड़ा", "सबसे छोटा"])

    if question_type == "सबसे बड़ा":
        answer = a
        reasoning = (
            f"{a} > {b} और {b} > {c}। "
            f"इसलिए संक्रमण नियम के अनुसार {a} > {c} "
            f"और {a} सबसे बड़ा है।"
        )
    else:
        answer = c
        reasoning = (
            f"{a} > {b} और {b} > {c}। "
            f"इसलिए {c} सबसे छोटा है।"
        )

    question = (
        f"यदि {a} का मान {b} से अधिक है और "
        f"{b} का मान {c} से अधिक है, "
        f"तो इनमें {question_type} कौन है?"
    )

    return make_record(
        record_id,
        "transitive_comparison",
        question,
        reasoning,
        answer,
    )


def equality_comparison(record_id):
    a, b = random.sample(NAMES, 2)
    relation, unit, relation_word = random.choice(RELATIONS)
    value = random.randint(5, 100)

    question = (
        f"{a} की {relation} {value} {unit} है। "
        f"{b} की {relation} भी {value} {unit} है। "
        f"किसकी {relation_word} अधिक है?"
    )

    reasoning = (
        f"{a} और {b} दोनों का मान {value} है। "
        "इसलिए दोनों समान हैं।"
    )

    return make_record(
        record_id,
        "equality",
        question,
        reasoning,
        "दोनों समान हैं",
    )


def multi_hop(record_id):
    a, b, c, d = random.sample(NAMES, 4)

    values = sorted(
        random.sample(range(10, 100), 4),
        reverse=True,
    )

    va, vb, vc, vd = values

    question = (
        f"चार व्यक्तियों के मान इस प्रकार हैं: "
        f"{a} > {b}, {b} > {c}, और {c} > {d}। "
        f"इनमें सबसे छोटा कौन है?"
    )

    reasoning = (
        f"{a} > {b}, {b} > {c}, और {c} > {d}। "
        f"इसलिए क्रम {a} > {b} > {c} > {d} है। "
        f"सबसे छोटा {d} है।"
    )

    return make_record(
        record_id,
        "multi_hop",
        question,
        reasoning,
        d,
    )


def basic_logic(record_id):
    a, b = random.sample(NAMES, 2)

    question = (
        f"यदि {a} की उम्र {b} से अधिक है, "
        f"तो {b} की उम्र {a} की तुलना में कैसी है?"
    )

    reasoning = (
        f"यदि {a} > {b}, तो उल्टे संबंध में {b} < {a} होगा।"
    )

    return make_record(
        record_id,
        "basic_logic",
        question,
        reasoning,
        "कम",
    )


def generate_dataset():
    generators = [
        (direct_comparison, 2000),
        (ordering, 2000),
        (transitive_comparison, 2500),
        (equality_comparison, 1500),
        (multi_hop, 1500),
        (basic_logic, 500),
    ]

    records = []
    record_id = 1

    for generator, count in generators:
        for _ in range(count):
            records.append(generator(record_id))
            record_id += 1

    random.shuffle(records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(f"Generated {len(records)} Hindi examples")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_dataset()
