import math

import torch
import torch.nn.functional as F


def generate(
    model,
    prompt_ids,
    max_new_tokens,
    temperature=1.0,
    device="cpu",
):
    """
    Autoregressively generate tokens from a prompt.

    temperature=0 gives greedy decoding.
    """

    model.eval()

    tokens = prompt_ids.clone().to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):

            context = tokens[:, -model.seq_len:]

            logits = model(context)

            next_logits = logits[:, -1, :]

            if temperature == 0:
                next_token = torch.argmax(
                    next_logits,
                    dim=-1,
                    keepdim=True,
                )

            else:
                next_logits = (
                    next_logits / temperature
                )

                probabilities = F.softmax(
                    next_logits,
                    dim=-1,
                )

                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                )

            tokens = torch.cat(
                [tokens, next_token],
                dim=1,
            )

    return tokens


def calculate_perplexity(loss):
    return math.exp(loss)


def calculate_bits_per_token(loss):
    return loss / math.log(2)


def calculate_bpb(loss, average_bytes_per_token):
    """
    Approximate bits-per-byte from bits/token.

    average_bytes_per_token should be computed from
    the evaluated corpus.
    """

    bits_per_token = calculate_bits_per_token(
        loss
    )

    return (
        bits_per_token
        / average_bytes_per_token
    )


def distinct_n(tokens, n):
    """
    Calculate Distinct-n.

    tokens: list of token IDs.
    """

    if len(tokens) < n:
        return 0.0

    ngrams = [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]

    return len(set(ngrams)) / len(ngrams)


def repetition_rate(tokens, n=3):
    """
    Fraction of n-grams that are repeated.
    """

    if len(tokens) < n:
        return 0.0

    ngrams = [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]

    unique = len(set(ngrams))

    return 1.0 - (
        unique / len(ngrams)
    )
