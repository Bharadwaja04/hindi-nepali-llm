import json
import unicodedata
from pathlib import Path

import pytest

from indic_lm.languages import LANGUAGES, REPO_ROOT
from indic_lm.tokenizer import Tokenizer

SENTENCES = {
    "hindi": ["भारत एक सुंदर देश है।", "आज सुबह दिल्ली में मौसम बहुत अच्छा है।", "शिक्षा के क्षेत्र में तकनीक"],
    "nepali": ["नेपाल सुन्दर देश हो।", "नेपालको राजधानी काठमाडौँ हो।", "कृषि क्षेत्रको विकास"],
}


def test_round_trip_tiny(tiny_tokenizer):
    for text in SENTENCES["hindi"] + SENTENCES["nepali"] + ["mixed English 123 text।", ""]:
        assert tiny_tokenizer.verify_round_trip(text)


def test_encoding_is_deterministic(tiny_tokenizer):
    text = SENTENCES["hindi"][0]
    assert tiny_tokenizer.encode(text) == tiny_tokenizer.encode(text)


def test_nfc_normalisation_makes_equivalent_forms_identical(tiny_tokenizer):
    composed = "क़"  # U+0958
    decomposed = unicodedata.normalize("NFD", composed)
    assert tiny_tokenizer.encode(composed) == tiny_tokenizer.encode(decomposed)


def test_merges_reduce_token_count(tiny_tokenizer):
    text = SENTENCES["hindi"][0]
    assert len(tiny_tokenizer.encode(text)) < len(text.encode("utf-8"))


def test_save_and_load_preserve_behaviour(tiny_tokenizer, tmp_path):
    path = tmp_path / "tok.json"
    tiny_tokenizer.save(path)
    loaded = Tokenizer.load(path)
    for text in SENTENCES["hindi"]:
        assert loaded.encode(text) == tiny_tokenizer.encode(text)
    assert json.loads(path.read_text(encoding="utf-8"))["format"] == "regex_bpe_v1"


def test_untrained_ids_map_to_bytes():
    tok = Tokenizer("hindi", vocab_size=256)
    assert tok.encode("a") == [97]
    assert tok.decode([97]) == "a"


def test_unknown_id_raises(tiny_tokenizer):
    with pytest.raises(ValueError):
        tiny_tokenizer.decode([10**6])


@pytest.mark.parametrize("language", ["hindi", "nepali"])
def test_committed_tokenizer_artifacts(language):
    """The released 31,744-token vocabularies: structure and round-trip."""
    path = REPO_ROOT / "artifacts" / "tokenizers" / f"{language}_bpe_31744.json"
    tok = Tokenizer.load(path)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["vocab_size"] == 31744
    assert data["base_vocab_size"] == 256
    assert len(data["merges"]) == 31744 - 256 == 31488
    assert len(tok.vocab) == 31744
    for text in SENTENCES[language]:
        assert tok.verify_round_trip(text)
        ids = tok.encode(text)
        assert max(ids) < 31744


def test_languages_registry_is_complete():
    assert set(LANGUAGES) == {"hindi", "nepali"}
    for spec in LANGUAGES.values():
        assert spec.model_config_path.exists()
        assert spec.tokenizer_path.exists()
