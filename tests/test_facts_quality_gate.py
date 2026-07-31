"""Tests for the PDF-artifact gate in `facts/quality.py`."""

from __future__ import annotations

from rag_gt.facts.quality import _looks_like_pdf_artifact, structural_quality


def test_good_fact_passes():
    text = "The temperature must not exceed 1500 C during the welding test."
    assert not _looks_like_pdf_artifact(text)
    assert structural_quality(text) > 0.5


def test_replacement_char_flagged():
    text = "RWTH Aachen University Universit�tsbibliothek users only."
    assert _looks_like_pdf_artifact(text)
    assert structural_quality(text) == 0.0


def test_cid_glyph_flagged():
    text = "This contains cid:104 and cid:105 references from the PDF."
    assert _looks_like_pdf_artifact(text)
    assert structural_quality(text) == 0.0


def test_reversed_watermark_flagged():
    # A literal watermark fragment lifted from an actual PDF.
    text = "Company name dellortnocnu seipoc detnirP page footer."
    assert _looks_like_pdf_artifact(text)
    assert structural_quality(text) == 0.0


def test_too_short_flagged():
    assert _looks_like_pdf_artifact("hi mom")
    assert structural_quality("hi mom") == 0.0


def test_low_letter_density_flagged():
    text = "1 2 3 / // | : ; 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
    assert _looks_like_pdf_artifact(text)
    assert structural_quality(text) == 0.0
