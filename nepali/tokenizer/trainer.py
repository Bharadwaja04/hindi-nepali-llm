from collections import Counter
from .bpe import merge
from .vocab import build_vocab

class BPETrainer:
    """
    Train byte-level BPE from scratch on regex-pre-tokenized text.
    """

    def __init__(self, config, pretokenizer):
        self.config = config
        self.pretokenizer = pretokenizer
        self.merges = {}
        self.vocab = build_vocab(
            self.merges,
            config.base_vocab_size
        )

    def train(self, text, verbose=True):
        self.config.validate()

        text = self.pretokenizer.normalize(text)
        pieces = self.pretokenizer.split(text)

        token_sequences = [
            list(piece.encode("utf-8"))
            for piece in pieces
            if piece
        ]

        num_merges = (
            self.config.vocab_size
            - self.config.base_vocab_size
        )

        self.merges = {}

        for i in range(num_merges):
            stats = Counter()

            for ids in token_sequences:
                stats.update(zip(ids, ids[1:]))

            if not stats:
                break

            pair = max(stats, key=stats.get)
            new_id = self.config.base_vocab_size + i

            self.merges[pair] = new_id

            if verbose:
                print(
                    f"merge {i + 1:>6}/{num_merges} | "
                    f"pair={pair} | token={new_id} | "
                    f"frequency={stats[pair]}"
                )

            token_sequences = [
                merge(ids, pair, new_id)
                for ids in token_sequences
            ]

        self.vocab = build_vocab(
            self.merges,
            self.config.base_vocab_size
        )

        return self.merges, self.vocab
