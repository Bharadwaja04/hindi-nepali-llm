"""Held-out evaluation of the reasoning fine-tune (exact-match on the test split).

The original project reported a 4-question notebook check for Hindi and a
60-question sample for Nepali. This module evaluates on the full held-out
``test.jsonl`` split and reports strict and lenient accuracy per category,
plus a repetition rate for detecting degenerate loops.

Answer extraction: generation is greedy and stops at the first newline
after the answer label (or at ``--max-new-tokens``). The predicted answer is
the text after the *last* answer label in the generated continuation.

    python -m indic_lm.evaluation.reasoning --language hindi \
        --checkpoint models/hindi_finetuned_phase3.pt --prompt-mode qa
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch

from indic_lm.evaluation.metrics import repetition_rate
from indic_lm.languages import LanguageSpec, get_language
from indic_lm.model.transformer import DecoderOnlyLM, load_state_dict_compat
from indic_lm.tokenizer import Tokenizer

_PUNCT = re.compile(r"[\s।॥.,;:!?\"'()\[\]{}\-_]+")


def load_records(path, limit: int | None = None) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def normalize_answer(text: str) -> str:
    """NFC, strip surrounding punctuation/danda/whitespace, collapse spaces."""
    text = unicodedata.normalize("NFC", text).strip()
    text = _PUNCT.sub(" ", text).strip()
    return text


def extract_answer(continuation: str, answer_label: str) -> str:
    """Text after the last ``<answer_label>:`` up to the end of that line."""
    marker = f"{answer_label}:"
    if marker in continuation:
        tail = continuation.rsplit(marker, 1)[1]
    else:
        tail = continuation
    return tail.split("\n", 1)[0].strip()


def is_exact(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def is_lenient(pred: str, gold: str) -> bool:
    """Gold answer appears as a whole token sequence in the predicted line."""
    p, g = normalize_answer(pred), normalize_answer(gold)
    return bool(g) and (p == g or f" {g} " in f" {p} ")


@torch.no_grad()
def generate_until(
    model: DecoderOnlyLM,
    tokenizer: Tokenizer,
    prompt: str,
    answer_label: str,
    max_new_tokens: int,
    device,
    stop_after_answer_line: bool = True,
) -> tuple[list[int], str]:
    """Greedy decoding; stop once the answer line (after ``answer_label:``) ends."""
    model.eval()
    ids = tokenizer.encode(prompt)
    generated: list[int] = []
    marker = f"{answer_label}:"

    for _ in range(max_new_tokens):
        context = torch.tensor([ids[-model.seq_len:]], dtype=torch.long, device=device)
        next_id = int(model(context)[0, -1].argmax())
        ids.append(next_id)
        generated.append(next_id)

        if stop_after_answer_line:
            text = tokenizer.decode(generated)
            if marker in prompt.rsplit("\n", 1)[-1]:
                # answer-only prompt: stop at the first newline after some content
                if "\n" in text.lstrip(" ") and text.strip():
                    break
            elif marker in text:
                tail = text.rsplit(marker, 1)[1]
                if "\n" in tail.lstrip(" ") and tail.strip():
                    break
    return generated, tokenizer.decode(generated)


def evaluate(
    model,
    tokenizer,
    records: list[dict],
    spec: LanguageSpec,
    prompt_mode: str = "qa",
    max_new_tokens: int = 40,
    device="cpu",
    keep_examples: int = 20,
) -> dict:
    per_cat = defaultdict(lambda: {"n": 0, "exact": 0, "lenient": 0, "rep": 0})
    total = {"n": 0, "exact": 0, "lenient": 0, "rep": 0}
    examples = []

    for rec in records:
        q, gold, cat = rec["question"], rec["answer"], rec.get("category", "unknown")
        prompt = spec.qa_prompt(q) if prompt_mode == "qa" else spec.cot_prompt(q)
        gen_ids, continuation = generate_until(
            model, tokenizer, prompt, spec.answer_label, max_new_tokens, device
        )
        pred = extract_answer(continuation, spec.answer_label) if prompt_mode == "qa" else (
            extract_answer(continuation, spec.answer_label)
            if f"{spec.answer_label}:" in continuation else ""
        )
        exact, lenient = is_exact(pred, gold), is_lenient(pred, gold)
        rep = repetition_rate(gen_ids, n=3) > 0.5 if len(gen_ids) >= 3 else False

        for bucket in (per_cat[cat], total):
            bucket["n"] += 1
            bucket["exact"] += int(exact)
            bucket["lenient"] += int(lenient)
            bucket["rep"] += int(rep)

        if len(examples) < keep_examples:
            examples.append({"id": rec.get("id"), "category": cat, "question": q, "gold": gold,
                             "prediction": pred, "exact": exact, "raw": continuation})

    def summarize(b):
        n = max(b["n"], 1)
        return {"n": b["n"], "exact_match": b["exact"] / n, "lenient_match": b["lenient"] / n,
                "repetition_rate": b["rep"] / n}

    return {
        "prompt_mode": prompt_mode,
        "max_new_tokens": max_new_tokens,
        "overall": summarize(total),
        "by_category": {c: summarize(b) for c, b in sorted(per_cat.items())},
        "examples": examples,
    }


def baselines(records: list[dict]) -> dict:
    """Reference points a model must beat to show it learned anything.

    majority:      always output the most frequent gold answer.
    informed_chance: pick uniformly among the known answer strings that
                   literally appear in the question (falls back to the whole
                   answer vocabulary). This is what a model that only
                   copies names from the question, without reasoning, gets.
    """
    from collections import Counter

    answers = Counter(normalize_answer(r["answer"]) for r in records)
    majority = answers.most_common(1)[0]
    vocab = list(answers)
    expected = 0.0
    for r in records:
        gold = normalize_answer(r["answer"])
        q = normalize_answer(r["question"])
        options = [a for a in vocab if a and a in q] or vocab
        if gold in options:
            expected += 1.0 / len(options)
    n = max(len(records), 1)
    return {
        "majority_answer": majority[0],
        "majority_exact_match": majority[1] / n,
        "informed_chance_exact_match": expected / n,
        "distinct_answers": len(vocab),
    }


def load_model(checkpoint: str | Path, config: dict, device) -> DecoderOnlyLM:
    model = DecoderOnlyLM.from_config({**config, "dropout": 0.0})
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    load_state_dict_compat(model, state, strict=True)
    return model.to(device).eval()


def main(argv=None):
    p = argparse.ArgumentParser(description="Held-out reasoning evaluation.")
    p.add_argument("--language", required=True, choices=["hindi", "nepali"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer")
    p.add_argument("--data", help="Test JSONL (default: data/reasoning/<language>/splits/test.jsonl).")
    p.add_argument("--prompt-mode", choices=["qa", "cot"], default="qa",
                   help="qa: 'प्रश्न: ..\\n\\nउत्तर:'  cot: 'प्रश्न: ..\\n\\nसमाधान:' (model writes reasoning then answer)")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--limit", type=int)
    p.add_argument("--out")
    args = p.parse_args(argv)

    spec = get_language(args.language)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, spec.load_model_config(), device)
    tokenizer = Tokenizer.load(args.tokenizer or spec.tokenizer_path)
    records = load_records(args.data or spec.reasoning_test_path, args.limit)

    result = evaluate(model, tokenizer, records, spec, args.prompt_mode, args.max_new_tokens, device)
    result["baselines"] = baselines(records)
    result.update({"language": args.language, "checkpoint": str(args.checkpoint)})
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
