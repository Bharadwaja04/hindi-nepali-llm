"""Shared data-collection / preprocessing / tokenizer utilities.

Language-agnostic modules used by both the Hindi (Model H) and Nepali
(Model L) pipelines.  Per-language behaviour is controlled by CLI flags
(``--lang``, ``--input``, ``--output``, ``--source-tag``).  Data,
tokenizers, vocabularies and weights are kept strictly separate on disk.
"""
