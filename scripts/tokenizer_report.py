"""Sanity report for the released tokenizers (``make tokenizer-check``).

For each language it checks lossless round-trips (including Latin, digits, emoji) and measures, on the first
``--limit`` held-out reasoning questions:

* fertility (tokens per whitespace word), characters/token, bytes/token,
* the share of tokens that are whitespace only (see docs/02-tokenizer.md),
* the cross-language penalty (each tokenizer applied to the other language's text).

Writes ``artifacts/tokenizers/tokenizer_report.json`` and exits non-zero if any round-trip fails.
The reasoning questions are templated text, so the numbers show direction, not corpus-level fertility.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

from indic_lm.languages import LANGUAGES, REPO_ROOT, get_language
from indic_lm.tokenizer import Tokenizer

EDGE_CASES = [
    "भारत एक सुंदर देश है।",
    "नेपालको राजधानी काठमाडौँ हो।",
    "AI मॉडल 2026 🙂",
    "  leading and   multiple   spaces\n\nnewlines\ttabs ",
    "क़ ख़ ग़",  # nukta forms: NFC-normalised on encode
    "",
]


def load_questions(language: str, limit: int) -> list[str]:
    path = get_language(language).reasoning_test_path
    texts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(json.loads(line)["question"])
            if len(texts) >= limit:
                break
    return texts


def measure(tok: Tokenizer, texts: list[str]) -> dict:
    tokens = words = chars = nbytes = space_tokens = 0
    for text in texts:
        ids = tok.encode(text)
        tokens += len(ids)
        words += len(text.split())
        chars += len(text)
        nbytes += len(text.encode("utf-8"))
        space_tokens += sum(1 for i in ids if tok.decode([i]).strip() == "")
    return {
        "texts": len(texts),
        "fertility": round(tokens / words, 3),
        "chars_per_token": round(chars / tokens, 3),
        "bytes_per_token": round(nbytes / tokens, 3),
        "whitespace_token_share": round(space_tokens / tokens, 3),
        "fertility_excluding_whitespace_tokens": round((tokens - space_tokens) / words, 3),
    }


def round_trip_ok(tok: Tokenizer, text: str) -> bool:
    return tok.decode(tok.encode(text)) == unicodedata.normalize("NFC", text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=400, help="questions per language")
    parser.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "tokenizers" / "tokenizer_report.json"))
    args = parser.parse_args(argv)

    tokenizers = {name: Tokenizer.load(spec.tokenizer_path) for name, spec in LANGUAGES.items()}
    texts = {name: load_questions(name, args.limit) for name in LANGUAGES}
    report, failures = {}, 0

    for name, tok in tokenizers.items():
        bad = [t for t in EDGE_CASES + texts[name] if not round_trip_ok(tok, t)]
        failures += len(bad)
        report[name] = {
            "vocab_size": len(tok.vocab),
            "learned_merges": len(tok.merges),
            "round_trip_failures": len(bad),
            "on_own_language": measure(tok, texts[name]),
            "on_other_language": {
                other: measure(tok, texts[other]) for other in LANGUAGES if other != name
            },
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for name, r in report.items():
        own = r["on_own_language"]
        print(f"{name:7s} round-trip failures {r['round_trip_failures']} | fertility {own['fertility']} "
              f"| chars/token {own['chars_per_token']} | whitespace tokens {own['whitespace_token_share']:.1%} "
              f"| fertility without them {own['fertility_excluding_whitespace_tokens']}")
        for other, m in r["on_other_language"].items():
            print(f"        on {other} text: fertility {m['fertility']}")
    print(f"wrote {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
