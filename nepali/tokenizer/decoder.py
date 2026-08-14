class Decoder:
    """Decode token IDs using the learned vocabulary."""

    def __init__(self, vocab, special_tokens=None):
        self.vocab = vocab
        self.special_by_id = {
            idx: token
            for token, idx in (special_tokens or {}).items()
        }

    def decode(self, ids):
        byte_buffer = bytearray()
        output = []

        def flush():
            if byte_buffer:
                output.append(
                    bytes(byte_buffer).decode(
                        "utf-8",
                        errors="replace"
                    )
                )
                byte_buffer.clear()

        for idx in ids:
            if idx in self.special_by_id:
                flush()
                output.append(self.special_by_id[idx])
            else:
                if idx not in self.vocab:
                    raise ValueError(
                        f"Unknown token ID: {idx}"
                    )
                byte_buffer.extend(self.vocab[idx])

        flush()
        return "".join(output)
