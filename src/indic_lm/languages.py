"""Per-language settings in one place.

The Hindi and Nepali pipelines share all code. What differs per language
(names, prompt labels, default artifact paths) lives here, so that no
module needs a ``hindi/`` and a ``nepali/`` copy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Repository root when running from a source checkout (src/indic_lm/languages.py).
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    tier: str  # "higher-resource" | "lower-resource"
    question_label: str
    reasoning_label: str
    answer_label: str

    @property
    def model_config_path(self) -> Path:
        return REPO_ROOT / "configs" / self.name / "model.json"

    @property
    def tokenizer_path(self) -> Path:
        return REPO_ROOT / "artifacts" / "tokenizers" / f"{self.name}_bpe_31744.json"

    @property
    def reasoning_test_path(self) -> Path:
        return REPO_ROOT / "data" / "reasoning" / self.name / "splits" / "test.jsonl"

    def load_model_config(self) -> dict:
        with self.model_config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def qa_prompt(self, question: str) -> str:
        """Prompt ending at the answer label (answer-only format)."""
        return f"{self.question_label}: {question}\n\n{self.answer_label}:"

    def cot_prompt(self, question: str) -> str:
        """Prompt ending at the reasoning label (reasoning-then-answer format)."""
        return f"{self.question_label}: {question}\n\n{self.reasoning_label}:"


# Both languages use Devanagari and share the same label words in the datasets.
LANGUAGES: dict[str, LanguageSpec] = {
    "hindi": LanguageSpec(
        name="hindi",
        tier="higher-resource",
        question_label="प्रश्न",
        reasoning_label="समाधान",
        answer_label="उत्तर",
    ),
    "nepali": LanguageSpec(
        name="nepali",
        tier="lower-resource",
        question_label="प्रश्न",
        reasoning_label="समाधान",
        answer_label="उत्तर",
    ),
}


def get_language(name: str) -> LanguageSpec:
    try:
        return LANGUAGES[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown language {name!r}; expected one of {sorted(LANGUAGES)}") from None
