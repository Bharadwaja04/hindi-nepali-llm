import json
from pathlib import Path

from .config import DEFAULT_PATTERN, TokenizerConfig
from .decoder import Decoder
from .encoder import Encoder
from .pretokenizer import RegexPreTokenizer
from .trainer import BPETrainer
from .vocab import build_vocab


class Tokenizer:
    """
    Main public tokenizer interface.

    This class composes:
        pretokenizer + trainer + encoder + decoder

    Hindi and Nepali should each have their own Tokenizer instance
    trained independently on their own corpus.
    """

    def __init__(
        self,
        language,
        vocab_size=32000,
        pattern=DEFAULT_PATTERN,
        special_tokens=None,
    ):
        self.config = TokenizerConfig(
            language=language,
            vocab_size=vocab_size,
            pattern=pattern,
            special_tokens=special_tokens or {},
        )

        self.pretokenizer = RegexPreTokenizer(pattern)
        self.merges = {}
        self.vocab = build_vocab(self.merges, self.config.base_vocab_size)

        self.encoder = Encoder(
            self.merges, self.pretokenizer, self.config.special_tokens
        )
        self.decoder = Decoder(self.vocab, self.config.special_tokens)

    def train(self, text, verbose=True, char_budget=80_000_000):
        trainer = BPETrainer(self.config, self.pretokenizer)

        self.merges, self.vocab = trainer.train(
            text,
            verbose=verbose,
            char_budget=char_budget,
        )

        self.encoder = Encoder(
            self.merges, self.pretokenizer, self.config.special_tokens
        )
        self.decoder = Decoder(self.vocab, self.config.special_tokens)

        return self

    def encode(self, text):
        return self.encoder.encode(text)

    def decode(self, ids):
        return self.decoder.decode(ids)

    def verify_round_trip(self, text):
        normalized = self.pretokenizer.normalize(text)
        return self.decode(self.encode(normalized)) == normalized

    def tokenization_stats(self, text):
        normalized = self.pretokenizer.normalize(text)
        ids = self.encode(normalized)
        chars = len(normalized)
        byte_count = len(normalized.encode("utf-8"))
        token_count = len(ids)

        return {
            "language": self.config.language,
            "vocab_size": self.config.vocab_size,
            "characters": chars,
            "utf8_bytes": byte_count,
            "tokens": token_count,
            "characters_per_token": (chars / token_count if token_count else 0.0),
            "bytes_per_token": (byte_count / token_count if token_count else 0.0),
        }

    def save(self, path):
        path = Path(path)

        data = {
            "format": "regex_bpe_v1",
            "language": self.config.language,
            "vocab_size": self.config.vocab_size,
            "base_vocab_size": self.config.base_vocab_size,
            "pattern": self.config.pattern,
            "special_tokens": self.config.special_tokens,
            "merges": [[p0, p1, idx] for (p0, p1), idx in self.merges.items()],
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(
            language=data["language"],
            vocab_size=int(data["vocab_size"]),
            pattern=data["pattern"],
            special_tokens={
                token: int(idx) for token, idx in data.get("special_tokens", {}).items()
            },
        )

        tokenizer.merges = {
            (int(p0), int(p1)): int(idx) for p0, p1, idx in data["merges"]
        }

        tokenizer.vocab = build_vocab(
            tokenizer.merges, tokenizer.config.base_vocab_size
        )

        tokenizer.encoder = Encoder(
            tokenizer.merges, tokenizer.pretokenizer, tokenizer.config.special_tokens
        )
        tokenizer.decoder = Decoder(tokenizer.vocab, tokenizer.config.special_tokens)

        return tokenizer
