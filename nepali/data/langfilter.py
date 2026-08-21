"""Lightweight language identification for Devanagari scripts.

We do not use a pretrained model (project rules forbid pretrained
artefacts).  Instead we use Unicode-block ratios, which is sufficient for
filtering obviously-wrong text in monolingual Devanagari corpora.
"""

import unicodedata


def _char_block(ch):
    """Return a coarse Unicode block label for a character."""
    cp = ord(ch)
    if 0x0900 <= cp <= 0x097F:
        return "devanagari"
    if 0x0980 <= cp <= 0x09FF:
        return "bengali"
    if 0x0A00 <= cp <= 0x0A7F:
        return "gurmukhi"
    if 0x0A80 <= cp <= 0x0AFF:
        return "gujarati"
    if 0x0B00 <= cp <= 0x0B7F:
        return "oriya"
    if 0x0B80 <= cp <= 0x0BFF:
        return "tamil"
    if 0x0C00 <= cp <= 0x0C7F:
        return "telugu"
    if 0x0C80 <= cp <= 0x0CFF:
        return "kannada"
    if 0x0D00 <= cp <= 0x0D7F:
        return "malayalam"
    if 0x0E00 <= cp <= 0x0E7F:
        return "thai"
    if 0x0600 <= cp <= 0x06FF:
        return "arabic"
    if 0x0041 <= cp <= 0x007A or 0x00C0 <= cp <= 0x024F:
        return "latin"
    if ch.isspace():
        return "space"
    if ch.isdigit() or 0x0660 <= cp <= 0x0669 or 0x0966 <= cp <= 0x096F:
        return "digit"
    if unicodedata.category(ch).startswith("P"):
        return "punct"
    return "other"


def script_ratios(text):
    """Return a dict mapping block labels to fraction of characters."""
    if not text:
        return {}
    counts = {}
    for ch in text:
        b = _char_block(ch)
        counts[b] = counts.get(b, 0) + 1
    n = len(text)
    return {k: v / n for k, v in counts.items()}


def is_devanagari(text, min_ratio=0.60):
    """True if Devanagari chars make up >= ``min_ratio`` of non-space chars."""
    letters = [c for c in text if not c.isspace()]
    if not letters:
        return False
    deva = sum(1 for c in letters if 0x0900 <= ord(c) <= 0x097F)
    return deva / len(letters) >= min_ratio


def looks_hindi(text, min_ratio=0.55):
    """Hindi filter: mostly Devanagari, allow some Latin/digits/punct."""
    return is_devanagari(text, min_ratio=min_ratio)


def looks_nepali(text, min_ratio=0.55):
    """Nepali filter: same script as Hindi; ratio threshold applied."""
    return is_devanagari(text, min_ratio=min_ratio)
