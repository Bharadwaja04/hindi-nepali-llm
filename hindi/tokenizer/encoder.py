from .bpe import get_stats, merge

class Encoder:
    """Encode text using learned BPE merges."""

    def __init__(self, merges, pretokenizer, special_tokens=None):
        self.merges = merges
        self.pretokenizer = pretokenizer
        self.special_tokens = special_tokens or {}

    def _encode_piece(self, piece):
        ids = list(piece.encode("utf-8"))

        while len(ids) >= 2:
            stats = get_stats(ids)

            applicable = [
                pair for pair in stats
                if pair in self.merges
            ]

            if not applicable:
                break

            pair = min(
                applicable,
                key=lambda p: self.merges[p]
            )

            ids = merge(
                ids,
                pair,
                self.merges[pair]
            )

        return ids

    def encode(self, text):
        text = self.pretokenizer.normalize(text)
        pieces = self.pretokenizer.split(text)

        ids = []

        for piece in pieces:
            if piece in self.special_tokens:
                ids.append(self.special_tokens[piece])
            else:
                ids.extend(self._encode_piece(piece))

        return ids
