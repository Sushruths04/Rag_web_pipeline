"""Production-path tests for span normalization."""

from rag_gt.core.types import Document, Fact
from rag_gt.spans.normalization import (
    _exact_match_span,
    _token_trigrams,
    find_fact_spans,
    tokenize_document,
)


def test_token_trigrams():
    assert "hel" in _token_trigrams("hello")
    assert "ell" in _token_trigrams("hello")
    assert "llo" in _token_trigrams("hello")


def test_exact_match():
    doc = Document(doc_id="test", text="The temperature must not exceed 1500 degrees.")
    doc_tokens = tokenize_document(doc)
    fact_tokens = ["temperature", "must", "not", "exceed"]
    start, end = _exact_match_span(doc_tokens, fact_tokens)
    assert start >= 0
    assert doc_tokens[start:end] == fact_tokens


def test_span_normalization_basic():
    doc = Document(
        doc_id="test",
        text=(
            "The welding temperature must not exceed 1500C. "
            "Exceeding this limit causes thermal deformation of the base material."
        ),
        doc_type="ISO_STANDARD",
    )
    chunks = [
        {
            "chunk_id": "test_c0000",
            "doc_id": "test",
            "text": doc.text,
            "sentences": [
                "The welding temperature must not exceed 1500C.",
                "Exceeding this limit causes thermal deformation of the base material.",
            ],
            "tokens": doc.text.split(),
            "char_start": 0,
            "char_end": len(doc.text),
        }
    ]
    doc_tokens = tokenize_document(doc)
    fact = Fact(
        fact_id="test_F0001",
        text="The welding temperature must not exceed 1500C.",
        role="rule",
    )
    facts = find_fact_spans(doc, doc_tokens, [fact], chunks)
    assert len(facts[0].supporting_spans) >= 1
    span = facts[0].supporting_spans[0]
    assert span.start_token >= 0
    assert span.end_token > span.start_token
    assert span.end_token <= len(doc_tokens)


def test_span_exclusive_end():
    doc = Document(doc_id="test", text="The temperature shall be monitored.")
    chunks = [
        {
            "chunk_id": "test_c0000",
            "doc_id": "test",
            "text": doc.text,
            "sentences": ["The temperature shall be monitored."],
            "tokens": doc.text.split(),
            "char_start": 0,
            "char_end": len(doc.text),
        }
    ]
    doc_tokens = tokenize_document(doc)
    fact = Fact(
        fact_id="test_F0001", text="The temperature shall be monitored.", role="rule"
    )
    facts = find_fact_spans(doc, doc_tokens, [fact], chunks)
    span = facts[0].supporting_spans[0]
    recovered = doc_tokens[span.start_token : span.end_token]
    assert len(recovered) > 0
