"""Front-matter detector: high recall on boilerplate, zero false positives on domain facts."""

from rag_gt.facts.front_matter import is_boilerplate


def test_flags_standards_and_publishing_boilerplate():
    boilerplate = [
        "The official English version of the document is the authoritative one.",
        "Member bodies have the right to representation on the technical committee.",
        "Attention is drawn to the possibility that identifying patent rights is required.",
        "The use of trade names in this document does not imply an endorsement.",
        "© ISO 2024 — all rights reserved.",
        "ISBN 978-1-937538-85-9",
        "The text of ISO 15609-1:2019 was adopted by CEN.",
        "This document was prepared by Technical Committee ISO/TC 44.",
    ]
    for text in boilerplate:
        assert is_boilerplate(text), text


def test_does_not_flag_domain_technical_facts():
    technical = [
        "Preheating temperature is classified as an essential variable.",
        "A router that implements flooding must detect whether a received LSP is newer.",
        "The six structural tokens are the brackets and braces and their counterparts.",
        "Decreasing the test force increases the scatter of the measurement results.",
        "If you knew the value of each action, solving the n-armed bandit would be trivial.",
        "The change effort can be estimated from the number of affected requirements.",
    ]
    for text in technical:
        assert not is_boilerplate(text), text


def test_empty_is_not_boilerplate():
    assert not is_boilerplate("")
    assert not is_boilerplate(None)  # type: ignore[arg-type]
