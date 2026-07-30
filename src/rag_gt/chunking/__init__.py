"""Chunking strategies dispatched by doc_type.

`NLP` is exposed for backward compatibility; prefer `_get_nlp()` so that
spaCy is loaded lazily on first use rather than at import time.
"""

from rag_gt.chunking.strategies import _get_nlp, chunk_document


def __getattr__(name):
    if name == "NLP":
        return _get_nlp()
    raise AttributeError(name)


__all__ = ["chunk_document", "NLP"]
