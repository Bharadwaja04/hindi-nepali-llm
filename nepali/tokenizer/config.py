from dataclasses import dataclass, field

DEFAULT_PATTERN = (
    r"\s+|"
    r"\p{L}[\p{M}\p{L}]*|"
    r"\p{N}+|"
    r"[^\p{L}\p{N}\s]"
)

@dataclass
class TokenizerConfig:
    language: str
    vocab_size: int = 32000
    base_vocab_size: int = 256
    pattern: str = DEFAULT_PATTERN
    special_tokens: dict = field(default_factory=dict)

    def validate(self):
        if self.vocab_size < self.base_vocab_size:
            raise ValueError(
                f"vocab_size must be >= {self.base_vocab_size}"
            )
