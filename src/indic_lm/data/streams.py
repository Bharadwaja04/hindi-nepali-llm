import itertools
import json

import torch


def jsonl_document_stream(paths, split=None):
    """
    Yield text documents from JSONL files.

    If split is None, all documents are used.
    Otherwise, only records matching that split are used.
    """
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)

                if split is not None:
                    if record.get("split") != split:
                        continue

                text = record.get("text", "")

                if text:
                    yield text


def budgeted_token_stream(
    document_stream,
    tokenizer,
    token_budget,
):
    """
    Tokenize documents until exactly token_budget tokens
    have been yielded, if enough data is available.
    """
    tokens_seen = 0

    for text in document_stream:
        ids = tokenizer.encode(text)

        remaining = token_budget - tokens_seen

        if remaining <= 0:
            break

        take = min(len(ids), remaining)

        for token in ids[:take]:
            yield token

        tokens_seen += take

        if tokens_seen >= token_budget:
            break


def combined_token_stream(
    manual_paths,
    nonmanual_paths,
    tokenizer,
    manual_budget,
    nonmanual_budget,
):
    """
    Produce the Phase 2 token stream.

    Manual corpus:
        all documents are used.

    Non-manual corpus:
        only records with split == "train" are used.
    """

    manual_documents = jsonl_document_stream(
        manual_paths,
        split=None,
    )

    yield from budgeted_token_stream(
        manual_documents,
        tokenizer,
        manual_budget,
    )

    nonmanual_documents = jsonl_document_stream(
        nonmanual_paths,
        split="train",
    )

    yield from budgeted_token_stream(
        nonmanual_documents,
        tokenizer,
        nonmanual_budget,
    )


def batch_stream(
    token_stream,
    batch_size,
    seq_len,
):
    """
    Create continuous next-token prediction batches.

    Each batch consumes batch_size * seq_len new source
    tokens and carries one boundary token from the
    previous batch.
    """
    stream = iter(token_stream)

    try:
        previous_token = next(stream)
    except StopIteration:
        return

    while True:
        new_tokens = list(
            itertools.islice(
                stream,
                batch_size * seq_len,
            )
        )

        if len(new_tokens) < batch_size * seq_len:
            break

        tokens = [previous_token] + new_tokens

        data = torch.tensor(
            tokens,
            dtype=torch.long,
        )

        x = data[:-1].view(
            batch_size,
            seq_len,
        )

        y = data[1:].view(
            batch_size,
            seq_len,
        )

        yield x, y

        previous_token = new_tokens[-1]
