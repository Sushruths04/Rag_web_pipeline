"""Hello-world smoke test for the production `src/rag_gt` path."""

import json

from rag_gt.core.types import Document, MSFS, QuestionGT
from rag_gt.facts.extraction import extract_candidate_facts
from rag_gt.ingestion.cleaning import clean_text
from rag_gt.profiling.profiler import profile_document
from rag_gt.chunking.strategies import chunk_document
from rag_gt.spans.normalization import find_fact_spans, tokenize_document
from rag_gt.storage.gt_io import _from_dict, _to_dict, load_gt, save_gt
from tests.helpers import cleanup_workspace_tmp, make_workspace_tmp

SAMPLE_TEXT = (
    "The welding temperature must not exceed 1500 degrees Celsius. "
    "Exceeding this limit causes thermal deformation of the base material. "
    "The base material shall be inspected after cooling."
)


def test_hello_world():
    doc = Document(
        doc_id="toy_doc", text=clean_text(SAMPLE_TEXT), doc_type="ISO_STANDARD"
    )

    profile = profile_document(doc)
    chunks = chunk_document(doc, profile, chunk_size=256, chunk_overlap=32)
    assert len(chunks) >= 1
    valid_chunk_ids = {chunk["chunk_id"] for chunk in chunks}

    facts = extract_candidate_facts(doc, chunks, llm=None)
    assert len(facts) >= 1

    doc_toks = tokenize_document(doc)
    facts = find_fact_spans(doc, doc_toks, facts, chunks)
    facts = [f for f in facts if f.supporting_spans]
    assert len(facts) >= 1

    q = QuestionGT(
        q_id="toy_doc_q000",
        question="Why must the welding temperature not exceed 1500 degrees Celsius?",
        gold_answer="Because exceeding 1500 degrees causes thermal deformation of the base material.",
        msfs_list=[MSFS("msfs_1", [facts[0].fact_id])],
        doc_ids=["toy_doc"],
        required_fact_ids=[facts[0].fact_id],
        difficulty_reasoning_depth=1,
        difficulty_semantic_distance="local",
        required_facts=[facts[0]],
    )

    tmp_path = make_workspace_tmp("hello_world")
    try:
        save_gt([q], "hello_world", out_dir=tmp_path)
        loaded = load_gt("hello_world", in_dir=tmp_path)

        assert len(loaded) == 1
        assert loaded[0].q_id == "toy_doc_q000"
        assert loaded[0].gold_answer.startswith("Because")
        assert loaded[0].required_facts
        for fact in loaded[0].required_facts:
            assert fact.supporting_spans
            for span in fact.supporting_spans:
                assert span.end_token > span.start_token
                assert span.doc_id == "toy_doc"
                assert span.chunk_id in valid_chunk_ids

        encoded = _to_dict(q)
        assert encoded["required_fact_ids"] == [facts[0].fact_id]
        assert encoded["required_facts"]
        assert encoded["required_facts"][0]["supporting_spans"]
        json.dumps(encoded)
    finally:
        cleanup_workspace_tmp(tmp_path)


def test_gt_loader_does_not_infer_self_containment_known_from_legacy_score():
    entry = {
        "q_id": "q1",
        "question": "What is stated?",
        "gold_answer": "Alpha is stated.",
        "msfs_list": [{"msfs_id": "m1", "fact_ids": ["F1"]}],
        "doc_ids": ["doc"],
        "required_fact_ids": ["F1"],
        "required_facts": [
            {
                "fact_id": "F1",
                "text": "Alpha is stated.",
                "role": "definition",
                "supporting_spans": [],
                "self_containment_score": 0.62,
            }
        ],
        "difficulty_reasoning_depth": 1,
        "difficulty_semantic_distance": "local",
    }

    loaded = _from_dict(entry)
    fact = loaded.required_facts[0]
    assert fact.self_containment_score == 0.62
    assert fact.self_containment_known is False


def test_gt_loader_preserves_explicit_self_containment_known():
    entry = {
        "q_id": "q1",
        "question": "What is stated?",
        "gold_answer": "Alpha is stated.",
        "msfs_list": [{"msfs_id": "m1", "fact_ids": ["F1"]}],
        "doc_ids": ["doc"],
        "required_fact_ids": ["F1"],
        "required_facts": [
            {
                "fact_id": "F1",
                "text": "Alpha is stated.",
                "role": "definition",
                "supporting_spans": [],
                "self_containment_score": 0.62,
                "self_containment_known": True,
            }
        ],
        "difficulty_reasoning_depth": 1,
        "difficulty_semantic_distance": "local",
    }

    loaded = _from_dict(entry)
    fact = loaded.required_facts[0]
    assert fact.self_containment_score == 0.62
    assert fact.self_containment_known is True
