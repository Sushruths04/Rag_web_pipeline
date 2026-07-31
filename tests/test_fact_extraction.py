"""Production-path tests for fact extraction."""

from rag_gt.core.types import Document
from rag_gt.facts.extraction import FACT_ID_PREFIX, extract_candidate_facts
from rag_gt.facts.quality import structural_quality
from rag_gt.facts.roles import detect_role


def test_structural_quality():
    assert structural_quality("The temperature must not exceed 1500C.") > 0.5
    assert structural_quality("short") < 0.5


def test_detect_role_rule():
    assert detect_role("The system shall be tested.") == "rule"


def test_detect_role_constraint():
    assert detect_role("The system must not fail.") == "constraint"


def test_detect_role_condition():
    assert detect_role("If the temperature rises, shut down.") == "condition"


def test_extract_facts_basic():
    doc = Document(
        doc_id="test",
        text=(
            "The welding temperature must not exceed 1500C. "
            "Exceeding this limit causes thermal deformation."
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
                "Exceeding this limit causes thermal deformation.",
            ],
            "tokens": doc.text.split(),
            "char_start": 0,
            "char_end": len(doc.text),
        }
    ]
    facts = extract_candidate_facts(doc, chunks, llm=None)
    assert len(facts) >= 1
    expected_prefix = f"test_{FACT_ID_PREFIX}"
    assert all(f.fact_id.startswith(expected_prefix) for f in facts)
