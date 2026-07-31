from __future__ import annotations

from rag_gt.graph.edge_classifier import _is_generic_relation_claim


def test_generic_relation_claim_rejected() -> None:
    assert _is_generic_relation_claim(
        "Fact B provides additional context for Fact A about reinforcement learning."
    )
    assert _is_generic_relation_claim(
        "Both facts describe the same concept in slightly different language."
    )
    assert _is_generic_relation_claim(
        "The first fact explains a rule while the second fact gives an example."
    )


def test_specific_relation_claim_allowed() -> None:
    assert not _is_generic_relation_claim(
        "TD error determines how eligibility traces change value estimates for recent states."
    )
