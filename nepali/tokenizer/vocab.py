def build_vocab(merges, base_vocab_size=256):
    """
    Build token_id -> bytes from the base byte vocabulary
    and learned BPE merges.
    """
    vocab = {
        idx: bytes([idx])
        for idx in range(base_vocab_size)
    }

    for (p0, p1), idx in merges.items():
        if p0 not in vocab or p1 not in vocab:
            raise ValueError(
                f"Invalid merge: ({p0}, {p1}) -> {idx}"
            )
        vocab[idx] = vocab[p0] + vocab[p1]

    return vocab
