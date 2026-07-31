from rag_gt.core.types import Fact, Span
from rag_gt.generation.chain_quality import chain_quality_gate


def _fact(fid: str, text: str, page: int = 1) -> Fact:
    return Fact(
        fact_id=fid,
        text=text,
        role="definition",
        supporting_spans=[
            Span(
                doc_id="doc",
                chunk_id=f"doc_c{page:04d}",
                start_token=0,
                end_token=10,
                page_start=page,
                page_end=page,
            )
        ],
    )


def test_chain_quality_accepts_connected_local_chain():
    facts = [
        _fact("F1", "If gamma is zero, a myopic agent maximizes immediate rewards.", 74),
        _fact("F2", "A myopic agent can maximize each immediate reward separately.", 74),
    ]

    assert chain_quality_gate(facts).accepted


def test_chain_quality_rejects_compact_header_artifact():
    facts = [
        _fact("F1", "PLANNING AND LEARNING WITH TABULAR METHODS covers tabular planning.", 228),
        _fact("F2", "212CHAPTER 8. PLANNING AND LEARNING WITH TABULAR METHODS", 226),
    ]

    verdict = chain_quality_gate(facts)

    assert not verdict.accepted
    assert verdict.reason.startswith("source_artifact:")


def test_chain_quality_rejects_far_page_stitched_chain():
    facts = [
        _fact("F1", "Algorithms can rely on the assumption of discrete states.", 223),
        _fact("F2", "A certainty-equivalence estimate is almost never feasible to compute directly.", 168),
    ]

    verdict = chain_quality_gate(facts, max_page_gap=40)

    assert not verdict.accepted
    assert verdict.reason == "source_gap_too_large"


def test_chain_quality_rejects_weak_context_fragments():
    facts = [
        _fact("F1", "The next choice is of the continuous variables with which to represent the state.", 298),
        _fact("F2", "Suppose we address a task with two continuous state variables.", 251),
    ]

    verdict = chain_quality_gate(facts)

    assert not verdict.accepted
    assert verdict.reason == "weak_context_fragment"


def test_chain_quality_rejects_duplicate_fact_loop():
    facts = [
        _fact("F1", "Policy iteration improves a policy using value estimates.", 10),
        _fact("F1", "Policy iteration improves a policy using value estimates.", 10),
    ]

    verdict = chain_quality_gate(facts)

    assert not verdict.accepted
    assert verdict.reason == "duplicate_fact_loop"
