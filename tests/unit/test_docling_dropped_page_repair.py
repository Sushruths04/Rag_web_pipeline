"""Regression tests for silent Docling page drops.

Root cause (2026-08-01, DIN EN ISO 13919-1-ENG): Docling swallows some
native-layer failures -- notably ``std::bad_alloc`` raised by its own
preprocessing stage -- and returns a *partial* document. convert() reports
success, so the caller's try/except never fires, and the affected pages
vanish with no signal in the output, the logs, or the gates. On that
standard, pages 18-20 (non-deterministically 17-20) were dropped, and they
held the imperfection-classification tables that are the document's entire
technical substance.

Fix: any page in a converted range that produced no source unit is
re-extracted via the PyMuPDF layout path and spliced back in page order,
with char offsets reassigned so the repaired range is structurally
identical to an intact one.
"""

from __future__ import annotations

from pathlib import Path

import rag_gt.ingestion.docling_pdf as dpdf
from rag_gt.core.types import SourceUnit
from rag_gt.ingestion.docling_pdf import (
    _effective_pages,
    _repair_dropped_pages,
    _reserialize,
)


def _u(text: str, page: int | None, start: int = 0, extractor: str = "docling") -> SourceUnit:
    return SourceUnit(
        doc_id="d",
        char_start=start,
        char_end=start + len(text),
        text=text,
        page_no=page,
        extractor=extractor,
    )


class TestEffectivePages:
    def test_carries_last_page_forward_for_prov_less_units(self):
        units = [_u("a", 1), _u("b", None), _u("c", 2), _u("d", None)]
        assert _effective_pages(units) == [1, 1, 2, 2]

    def test_leading_prov_less_unit_sorts_to_front_not_last_page(self):
        units = [_u("a", None), _u("b", 5)]
        assert _effective_pages(units) == [0, 5]


class TestReserialize:
    def test_offsets_match_the_joined_text(self):
        units = [_u("alpha", 1), _u("beta", 2), _u("gamma", 3)]
        text, out = _reserialize(units, cursor=0)
        assert text == "alpha\n\nbeta\n\ngamma"
        for u in out:
            assert text[u.char_start:u.char_end] == u.text

    def test_offsets_are_based_at_the_supplied_cursor(self):
        text, out = _reserialize([_u("alpha", 1), _u("beta", 2)], cursor=100)
        assert out[0].char_start == 100
        assert out[1].char_start == 100 + len("alpha") + 2

    def test_preserves_page_and_extractor_metadata(self):
        _, out = _reserialize([_u("x", 7, extractor="pymupdf")], cursor=0)
        assert out[0].page_no == 7
        assert out[0].extractor == "pymupdf"


class TestRepairDroppedPages:
    def test_missing_middle_pages_are_recovered_and_spliced_in_page_order(
        self, monkeypatch
    ):
        """Docling returned pages 1,2,5 of a 1-5 range; 3 and 4 vanished."""
        page_units = [_u("p1", 1), _u("p2", 2), _u("p5", 5)]
        page_text = "p1\n\np2\n\np5"

        def fake_legacy(path, doc_id, source_sha1="", page_range=None):
            p = page_range[0]
            return f"recovered{p}", [_u(f"recovered{p}", p, extractor="pymupdf")]

        monkeypatch.setattr("rag_gt.ingestion.pdf.extract_pdf_with_layout", fake_legacy)

        text, units = _repair_dropped_pages(
            page_text, page_units,
            source=Path("doc.pdf"), doc_id="d", source_sha1="",
            page_range=(1, 5), cursor=0,
        )

        assert [u.page_no for u in units] == [1, 2, 3, 4, 5], "must splice in page order"
        assert text == "p1\n\np2\n\nrecovered3\n\nrecovered4\n\np5"
        for u in units:
            assert text[u.char_start:u.char_end] == u.text, "offsets must stay consistent"

    def test_no_missing_pages_is_a_passthrough(self, monkeypatch):
        page_units = [_u("p1", 1), _u("p2", 2)]

        def boom(*a, **k):
            raise AssertionError("must not attempt repair when nothing is missing")

        monkeypatch.setattr("rag_gt.ingestion.pdf.extract_pdf_with_layout", boom)

        text, units = _repair_dropped_pages(
            "p1\n\np2", page_units,
            source=Path("doc.pdf"), doc_id="d", source_sha1="",
            page_range=(1, 2), cursor=0,
        )
        assert text == "p1\n\np2"
        assert units is page_units

    def test_genuinely_blank_pages_are_self_correcting(self, monkeypatch):
        """A blank page yields no units; output must be unchanged, not corrupted."""
        page_units = [_u("p1", 1)]

        def fake_blank(path, doc_id, source_sha1="", page_range=None):
            return "", []

        monkeypatch.setattr("rag_gt.ingestion.pdf.extract_pdf_with_layout", fake_blank)

        text, units = _repair_dropped_pages(
            "p1", page_units,
            source=Path("doc.pdf"), doc_id="d", source_sha1="",
            page_range=(1, 3), cursor=0,
        )
        assert text == "p1"
        assert units is page_units

    def test_repair_failure_on_one_page_does_not_lose_the_others(self, monkeypatch):
        page_units = [_u("p1", 1)]
        calls = []

        def flaky(path, doc_id, source_sha1="", page_range=None):
            p = page_range[0]
            calls.append(p)
            if p == 2:
                raise RuntimeError("page 2 unreadable")
            return f"ok{p}", [_u(f"ok{p}", p, extractor="pymupdf")]

        monkeypatch.setattr("rag_gt.ingestion.pdf.extract_pdf_with_layout", flaky)

        text, units = _repair_dropped_pages(
            "p1", page_units,
            source=Path("doc.pdf"), doc_id="d", source_sha1="",
            page_range=(1, 3), cursor=0,
        )

        assert calls == [2, 3]
        assert [u.page_no for u in units] == [1, 3], "page 3 must survive page 2's failure"
        for u in units:
            assert text[u.char_start:u.char_end] == u.text

    def test_repaired_units_keep_offsets_valid_at_a_nonzero_cursor(self, monkeypatch):
        page_units = [_u("p1", 1, start=500)]

        def fake_legacy(path, doc_id, source_sha1="", page_range=None):
            p = page_range[0]
            return f"r{p}", [_u(f"r{p}", p, extractor="pymupdf")]

        monkeypatch.setattr("rag_gt.ingestion.pdf.extract_pdf_with_layout", fake_legacy)

        text, units = _repair_dropped_pages(
            "p1", page_units,
            source=Path("doc.pdf"), doc_id="d", source_sha1="",
            page_range=(1, 2), cursor=500,
        )

        assert units[0].char_start == 500
        for u in units:
            assert text[u.char_start - 500:u.char_end - 500] == u.text


class _FakeBBox:
    l, t, r, b = 0.0, 0.0, 1.0, 1.0
    coord_origin = type("O", (), {"value": "BOTTOMLEFT"})()


class _FakeProv:
    def __init__(self, page_no):
        self.page_no = page_no
        self.bbox = _FakeBBox()


class _FakeItem:
    def __init__(self, text, page_no):
        self.text = text
        self.prov = [_FakeProv(page_no)]
        self.self_ref = f"#/texts/{text}"


class _FakeDoc:
    """Mimics a partial Docling document: only page 1 survived."""
    def iterate_items(self):
        yield _FakeItem("page one body", 1), 0


class TestRepairIsWiredIntoTheConverter:
    """Guards the call site: deleting the repair hook must fail a test.

    The 10 unit tests above all pass against an unwired _repair_dropped_pages,
    so without this the regression could silently return.
    """

    def test_silently_dropped_pages_are_repaired_end_to_end(self, monkeypatch, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        monkeypatch.setattr(dpdf, "_page_ranges", lambda source, size: [(1, 3)])

        class _FakeConverter:
            def __init__(self, *a, **k):
                pass

            def convert(self, source, raises_on_error=True, page_range=None):
                # Succeeds while returning only page 1 -- exactly the bad_alloc shape.
                return type("R", (), {"document": _FakeDoc()})()

        monkeypatch.setattr(
            "docling.document_converter.DocumentConverter", _FakeConverter
        )

        def fake_legacy(path, doc_id, source_sha1="", page_range=None):
            p = page_range[0]
            return f"recovered page {p}", [
                SourceUnit(
                    doc_id=doc_id, char_start=0, char_end=16,
                    text=f"recovered page {p}", page_no=p, extractor="pymupdf",
                )
            ]

        monkeypatch.setattr("rag_gt.ingestion.pdf.extract_pdf_with_layout", fake_legacy)

        text, units = dpdf.extract_pdf_docling_with_layout(
            str(pdf), "d", do_table_structure=True,
            min_text_chars=0, min_text_file_ratio=0.0,
        )

        assert sorted({u.page_no for u in units}) == [1, 2, 3], (
            "pages 2-3 were silently dropped by the converter and must be repaired"
        )
        assert "recovered page 2" in text and "recovered page 3" in text
        for u in units:
            assert text[u.char_start:u.char_end] == u.text
