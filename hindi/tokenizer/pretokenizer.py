import unicodedata
import regex

class RegexPreTokenizer:
    """
    Unicode-aware regex pre-tokenizer.

    It only decides the text pieces on which BPE operates.
    It does not contain a pretrained vocabulary or merge table.
    """

    def __init__(self, pattern):
        self.pattern = pattern
        self.compiled_pattern = regex.compile(
            pattern,
            flags=regex.UNICODE
        )

    @staticmethod
    def normalize(text):
        return unicodedata.normalize("NFC", text)

    def split(self, text):
        text = self.normalize(text)
        return [
            match.group(0)
            for match in self.compiled_pattern.finditer(text)
        ]
