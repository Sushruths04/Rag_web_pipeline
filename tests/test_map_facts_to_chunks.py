from pathlib import Path

from rag_gt.cli.map_facts_to_chunks import (
    _chunks_by_doc,
    _map_fact,
    _question_map,
    _summary,
)
from rag_gt.core.types import Fact, MSFS, QuestionGT, Span


def _question() -> QuestionGT:
    fact = Fact(
        fact_id="doc_F1",
        text="Alpha beta gamma is long enough.",
        role="definition",
        supporting_spans=[
            Span(
                doc_id="doc",
                chunk_id="old_c000000",
                start_token=0,
                end_token=4,
                char_start=10,
                char_end=30,
                page_start=2,
                page_end=2,
                source_sha1="src-sha",
            )
        ],
    )
    return QuestionGT(
        q_id="doc_q001",
        question="What follows from alpha?",
        gold_answer="Alpha beta gamma.",
        msfs_list=[MSFS(msfs_id="m1", fact_ids=[fact.fact_id])],
        doc_ids=["doc"],
        required_fact_ids=[fact.fact_id],
        difficulty_reasoning_depth=1,
        difficulty_semantic_distance="local",
        required_facts=[fact],
    )


def test_map_fact_to_external_512_chunk_by_source_overlap():
    q = _question()
    chunks = _chunks_by_doc(
        [
            {
                "chunk_id": "user512_c001",
                "doc_id": "doc",
                "text": "before",
                "char_start": 0,
                "char_end": 9,
            },
            {
                "chunk_id": "user512_c002",
                "doc_id": "doc",
                "text": "Alpha beta gamma",
                "char_start": 9,
                "char_end": 60,
                "page_start": 2,
                "page_end": 2,
            },
        ]
    )

    row = _map_fact(
        q,
        q.required_facts[0],
        chunks,
        min_overlap_ratio=0.0,
        fallback_fuzzy_threshold=85.0,
    )

    assert row["fact_id"] == "doc_F1"
    assert row["best_chunk_id"] == "user512_c002"
    assert row["matched_chunk_ids"] == ["user512_c002"]
    assert row["best_overlap_ratio"] == 1.0
    assert row["mapping_method"] == "source_sha1_char_overlap"


def test_map_fact_falls_back_to_page_fuzzy_when_offsets_missing():
    q = _question()
    chunks = _chunks_by_doc(
        [
            {
                "chunk_id": "user_page_c001",
                "doc_id": "doc",
                "text": "Alpha beta gamma is long enough.",
                "page_start": 2,
                "page_end": 2,
            }
        ]
    )

    row = _map_fact(
        q,
        q.required_facts[0],
        chunks,
        min_overlap_ratio=0.0,
        fallback_fuzzy_threshold=85.0,
    )

    assert row["best_chunk_id"] == "user_page_c001"
    assert row["mapping_method"] == "page_fuzzy_text"


def test_question_map_marks_unmapped_facts():
    q = _question()
    fact_rows = [
        {
            "fact_id": "doc_F1",
            "matched_chunk_ids": [],
        }
    ]

    row = _question_map(q, fact_rows, "char1024_overlap128")

    assert row["chunk_profile_id"] == "char1024_overlap128"
    assert row["mapping_complete"] is False
    assert row["unmapped_fact_ids"] == ["doc_F1"]


def test_mapping_summary_reports_coverage():
    q = _question()
    fact_rows = [
        {
            "fact_id": "doc_F1",
            "matched_chunk_ids": ["c1"],
            "mapping_method": "doc_id_char_overlap",
        }
    ]
    question_rows = [
        {
            "q_id": "doc_q001",
            "mapping_complete": True,
            "required_chunk_ids": ["c1"],
        }
    ]

    summary = _summary(
        gt_path=Path("gt.jsonl"),
        chunks_path=Path("chunks.jsonl"),
        chunk_profile_id="char512_overlap64",
        questions=[q],
        chunks=[{"chunk_id": "c1"}],
        fact_rows=fact_rows,
        question_rows=question_rows,
    )

    assert summary["fact_mapping_coverage"] == 1.0
    assert summary["question_mapping_coverage"] == 1.0
    assert summary["mapping_methods"] == {"doc_id_char_overlap": 1}
