"""Span normalization: map facts back to document tokens with fuzzy fallback."""

from rag_gt.spans.normalization import find_fact_spans, tokenize_document

__all__ = ["find_fact_spans", "tokenize_document"]
