from rag_gt.core.types import Document, Fact, SourceBBox, SourceUnit, Span
from rag_gt.source_mapping import annotate_chunk_with_source, map_fact_to_chunks
from rag_gt.spans.normalization import find_fact_spans, tokenize_document


def test_fact_span_carries_source_locator():
    text = "Alpha beta gamma.\n\nDelta epsilon zeta."
    doc = Document(
        doc_id="doc",
        text=text,
        source_path="source.pdf",
        source_sha1="abc",
        source_backend="docling",
        source_units=[
            SourceUnit(
                doc_id="doc",
                char_start=0,
                char_end=17,
                text="Alpha beta gamma.",
                page_no=1,
                block_id="p1_b1",
                paragraph_id="p1_b1",
                bboxes=[SourceBBox(page_no=1, l=1, t=10, r=100, b=1)],
            )
        ],
    )
    chunks = [
        {
            "chunk_id": "doc_c000000",
            "doc_id": "doc",
            "text": text,
            "char_start": 0,
            "char_end": len(text),
        }
    ]
    fact = Fact(fact_id="doc_F001", text="Alpha beta gamma.", role="definition")

    facts = find_fact_spans(doc, tokenize_document(doc), [fact], chunks)

    span = facts[0].supporting_spans[0]
    assert span.char_start == 0
    assert span.char_end == len("Alpha beta gamma.")
    assert span.page_start == 1
    assert span.page_end == 1
    assert span.block_ids == ["p1_b1"]
    assert span.bboxes[0].page_no == 1


def test_fact_to_chunk_mapping_survives_changed_chunk_ids():
    span = Span(
        doc_id="doc",
        chunk_id="old_c000000",
        start_token=0,
        end_token=3,
        char_start=6,
        char_end=16,
    )
    chunks = [
        {"chunk_id": "new_c000000", "doc_id": "doc", "char_start": 0, "char_end": 5},
        {"chunk_id": "new_c000001", "doc_id": "doc", "char_start": 5, "char_end": 20},
    ]

    assert map_fact_to_chunks([span], chunks) == ["new_c000001"]


def test_chunk_annotation_adds_page_coverage():
    doc = Document(
        doc_id="doc",
        text="One paragraph. Two paragraph.",
        source_units=[
            SourceUnit(
                doc_id="doc",
                char_start=0,
                char_end=14,
                page_no=3,
                block_id="p3_b1",
                paragraph_id="p3_b1",
                bboxes=[],
            )
        ],
        source_path="source.pdf",
        source_sha1="abc",
        source_backend="docling",
    )
    chunk = {"chunk_id": "doc_c000000", "doc_id": "doc", "char_start": 0, "char_end": 10}

    annotate_chunk_with_source(doc, chunk)

    assert chunk["page_start"] == 3
    assert chunk["page_end"] == 3
    assert chunk["source_path"] == "source.pdf"
    assert chunk["extractor"] == "docling"
