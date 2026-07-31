"""Regression tests for the docling_page_cap truncation bug.

Root cause (found investigating DIN EN ISO 13919-1-ENG, 2026-08-01): the
autonomous orchestrators (run.py, pipeline.py, eval_orchestrator.py) each
redeclared docling_page_cap with a default of 8, while ingest_document()'s
own reasoned default is 40/60. For docling_table documents over the cap,
ingest_document sliced the PDF down to a page-1..cap window and still ran
Docling on that slice -- silently dropping every page past the cap with no
signal reaching the caller. A 24-page standard with all-front-matter pages
1-8 therefore only ever ingested its title page / foreword / boilerplate;
the actual technical clauses (pages 9-24) were never extracted.

Fix: over-cap docling_table documents now fall back to legacy extraction of
the FULL document (matching ingest_document's own docstring contract)
instead of truncating, and every orchestrator default was raised to 60 so
normal-sized table-dense documents get real Docling table extraction
instead of hitting the fallback at all.
"""

from __future__ import annotations

import inspect

import pytest

from rag_gt.allpdf import eval_orchestrator, pipeline, run
from rag_gt.allpdf.ingest import ingest_document
from rag_gt.allpdf.preflight import BACKEND_DOCLING_TABLE, BACKEND_LEGACY, DocProfile
from rag_gt.core.types import SourceUnit


def _make_profile(page_count: int, backend: str = BACKEND_DOCLING_TABLE) -> DocProfile:
    return DocProfile(
        doc_id="din_iso_13919_1",
        path="fake.pdf",
        source_sha256="deadbeef",
        page_count=page_count,
        sampled_pages=min(page_count, 15),
        digital_text_ratio=0.9,
        scanned=False,
        mixed=False,
        avg_chars_per_page=1200.0,
        figure_density=0.1,
        table_density=0.6,
        multi_column=False,
        reading_order_risk="low",
        recommended_backend=backend,
    )


def _units(n: int) -> list[SourceUnit]:
    return [
        SourceUnit(doc_id="d", char_start=i, char_end=i + 1, text="x", page_no=i + 1)
        for i in range(n)
    ]


class TestOverCapFallsBackToLegacyFullDocument:
    def test_over_cap_docling_table_uses_legacy_on_full_page_range(self, monkeypatch):
        """24-page doc, cap=8: must NOT slice-and-still-run-docling on pages 1-8."""
        profile = _make_profile(page_count=24)
        docling_calls = []
        legacy_calls = []

        def fake_docling(*args, **kwargs):
            docling_calls.append((args, kwargs))
            return "docling text", _units(8)

        def fake_legacy(path, doc_id, source_sha1=None, page_range=None):
            legacy_calls.append(page_range)
            return "legacy text", _units(24)

        monkeypatch.setattr(
            "rag_gt.ingestion.docling_pdf.extract_pdf_docling_with_layout", fake_docling
        )
        monkeypatch.setattr("rag_gt.ingestion.pdf.extract_pdf_with_layout", fake_legacy)

        result = ingest_document(profile, docling_page_cap=8)

        assert docling_calls == [], "docling must not run at all on the over-cap path"
        assert legacy_calls == [None], "legacy must be called on the FULL doc (page_range=None), not a window"
        assert result.backend_used == BACKEND_LEGACY
        assert result.pages_covered == 24
        assert any("falling back to legacy" in n for n in result.notes)

    def test_within_cap_docling_table_still_uses_docling(self, monkeypatch):
        """24-page doc, cap=60: should get real Docling table extraction."""
        profile = _make_profile(page_count=24)
        docling_calls = []

        def fake_docling(*args, **kwargs):
            docling_calls.append((args, kwargs))
            return "docling text", _units(24)

        monkeypatch.setattr(
            "rag_gt.ingestion.docling_pdf.extract_pdf_docling_with_layout", fake_docling
        )

        result = ingest_document(profile, docling_page_cap=60)

        assert len(docling_calls) == 1
        assert result.backend_used == BACKEND_DOCLING_TABLE
        assert result.pages_covered == 24

    def test_huge_report_falls_back_rather_than_losing_136_pages(self, monkeypatch):
        """144-page report, default cap: must not truncate to page 1-N."""
        profile = _make_profile(page_count=144)
        docling_calls = []
        legacy_calls = []

        def fake_docling(*args, **kwargs):
            docling_calls.append((args, kwargs))
            return "docling text", _units(60)

        def fake_legacy(path, doc_id, source_sha1=None, page_range=None):
            legacy_calls.append(page_range)
            return "legacy text", _units(144)

        monkeypatch.setattr(
            "rag_gt.ingestion.docling_pdf.extract_pdf_docling_with_layout", fake_docling
        )
        monkeypatch.setattr("rag_gt.ingestion.pdf.extract_pdf_with_layout", fake_legacy)

        result = ingest_document(profile, docling_page_cap=60)

        assert docling_calls == []
        assert legacy_calls == [None]
        assert result.pages_covered == 144


class TestOrchestratorDefaultsAreNotTheOldUnsafe8:
    """Locks in the raised defaults so no wrapper silently regresses to 8."""

    @pytest.mark.parametrize(
        "fn",
        [
            run.run_pipeline,
            pipeline.run_doc_pipeline,
            eval_orchestrator.run_and_evaluate,
            eval_orchestrator.run_corpus,
        ],
    )
    def test_default_is_60_not_8(self, fn):
        default = inspect.signature(fn).parameters["docling_page_cap"].default
        assert default == 60, (
            f"{fn.__module__}.{fn.__qualname__} defaults docling_page_cap to "
            f"{default}; any docling_table doc over this many pages silently "
            f"loses its tail unless ingest_document's legacy fallback covers it"
        )
