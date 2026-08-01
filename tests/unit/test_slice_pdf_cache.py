"""Regression test for _slice_pdf's content-blind cache key.

Found 2026-08-01 while auditing the ingestion path. _slice_pdf caches a
page slice under sha1("{path}:{start}:{end}") only. The key says nothing
about the file's *contents*, so replacing or editing the PDF at that path
-- re-exporting a standard, dropping in a corrected scan, overwriting a
temp filename -- silently returns the previous document's pages. The
pipeline would then extract, chunk, and generate ground truth from a
document that is no longer on disk, with every provenance field pointing
at the current file. Silent and near-undetectable downstream.
"""

from __future__ import annotations

import fitz
import pytest

from rag_gt.allpdf.ingest import _slice_pdf


def _write_pdf(path, pages: list[str]) -> None:
    doc = fitz.open()
    for body in pages:
        page = doc.new_page()
        page.insert_text((72, 72), body)
    doc.save(str(path))
    doc.close()


def _text_of(path) -> str:
    doc = fitz.open(str(path))
    try:
        return "\n".join(p.get_text() for p in doc)
    finally:
        doc.close()


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    # _slice_pdf caches under data_dir(), which honours RAG_GT_DATA_DIR.
    # Point it at tmp_path so tests never touch the repo's real cache.
    monkeypatch.setenv("RAG_GT_DATA_DIR", str(tmp_path / "data"))


def test_replacing_the_source_pdf_does_not_serve_a_stale_slice(tmp_path):
    src = tmp_path / "standard.pdf"
    _write_pdf(src, ["ORIGINAL page one", "ORIGINAL page two"])

    first = _slice_pdf(str(src), (1, 2))
    assert "ORIGINAL" in _text_of(first)

    # Same path, different document (e.g. a corrected re-export).
    _write_pdf(src, ["REVISED page one", "REVISED page two"])

    second = _slice_pdf(str(src), (1, 2))
    assert "REVISED" in _text_of(second), (
        "slice cache served the previous document's pages after the source "
        "file changed; ground truth would be built from a stale document"
    )


def test_identical_file_still_hits_the_cache(tmp_path):
    src = tmp_path / "doc.pdf"
    _write_pdf(src, ["stable page"])

    a = _slice_pdf(str(src), (1, 1))
    b = _slice_pdf(str(src), (1, 1))
    assert a == b, "unchanged input must reuse the cached slice, not re-slice"


def test_different_ranges_get_different_slices(tmp_path):
    src = tmp_path / "doc.pdf"
    _write_pdf(src, ["page one text", "page two text", "page three text"])

    one = _slice_pdf(str(src), (1, 1))
    two = _slice_pdf(str(src), (2, 2))
    assert one != two
    assert "one" in _text_of(one)
    assert "two" in _text_of(two)


def test_cache_location_is_independent_of_the_working_directory(tmp_path, monkeypatch):
    """A bare relative "data/cache/..." scattered caches per launch directory."""
    src = tmp_path / "doc.pdf"
    _write_pdf(src, ["only page"])

    elsewhere = tmp_path / "some" / "other" / "cwd"
    elsewhere.mkdir(parents=True)

    monkeypatch.chdir(tmp_path)
    from_here = _slice_pdf(str(src), (1, 1))

    monkeypatch.chdir(elsewhere)
    from_there = _slice_pdf(str(src), (1, 1))

    assert from_here == from_there, (
        "slice cache moved with the working directory; the same slice would be "
        "re-cut per launch dir and stray data/ trees would accumulate"
    )
