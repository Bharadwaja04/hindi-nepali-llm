from collections import Counter

def get_stats(ids):
    """Count consecutive token pairs."""
    return Counter(zip(ids, ids[1:]))

def merge(ids, pair, idx):
    """Replace consecutive occurrences of pair with idx."""
    i = 0
    new_ids = []

    while i < len(ids):
        if (
            i < len(ids) - 1
            and ids[i] == pair[0]
            and ids[i + 1] == pair[1]
        ):
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1

    return new_ids
