"""Stage 1 — Adaptive Ingestion.

Routes each document to an extractor backend chosen by its Stage-0 profile:
- legacy  -> PyMuPDF block extraction (fast, bbox-rich) for clean digital PDFs
- docling_table -> Docling with table-structure for table-dense PDFs
- docling_ocr   -> Docling with OCR for scanned/image PDFs

Every emitted SourceUnit keeps page + bbox + char span + source hash + the
extractor name. Units are tagged with front/back-matter membership using the
Stage-0 detected ranges, replacing the hardcoded per-filename page excludes.

Robust by design: a Docling failure or oversized document falls back to legacy
extraction with a recorded note, so the autonomous pipeline never hard-stops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from loguru import logger

from rag_gt.allpdf.preflight import (
    BACKEND_DOCLING_OCR,
    BACKEND_DOCLING_TABLE,
    BACKEND_LEGACY,
    DocProfile,
)
from rag_gt.core.types import SourceUnit


@dataclass
class IngestResult:
    doc_id: str
    backend_used: str
    text: str
    units: List[SourceUnit]
    char_count: int
    n_units: int
    pages_covered: int
    front_matter_units: int
    back_matter_units: int
    notes: List[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "backend_used": self.backend_used,
            "char_count": self.char_count,
            "n_units": self.n_units,
            "pages_covered": self.pages_covered,
            "front_matter_units": self.front_matter_units,
            "back_matter_units": self.back_matter_units,
            "notes": self.notes,
        }


def _tag_matter(units: List[SourceUnit], profile: DocProfile) -> Tuple[int, int]:
    front = set(profile.front_matter_pages)
    back = set(profile.back_matter_pages)
    f = b = 0
    for u in units:
        # Reuse paragraph_id-free tagging via block_id suffix is fragile; use page.
        if u.page_no in front:
            f += 1
        elif u.page_no in back:
            b += 1
    return f, b


def _legacy(path: str, doc_id: str, sha: str, page_range) -> Tuple[str, List[SourceUnit]]:
    from rag_gt.ingestion.pdf import extract_pdf_with_layout

    return extract_pdf_with_layout(path, doc_id, source_sha1=sha, page_range=page_range)


def ingest_document(
    profile: DocProfile,
    *,
    page_range: Optional[Tuple[int, int]] = None,
    allow_docling: bool = True,
    docling_page_cap: int = 40,
) -> IngestResult:
    """Ingest one document according to its profile.

    ``docling_page_cap`` bounds Docling on large documents: beyond it, Docling
    (which is slow and memory-heavy) is skipped in favour of legacy extraction
    unless an explicit ``page_range`` is supplied. This keeps autonomous runs
    bounded while exercising the same code path on smaller documents.
    """
    path, doc_id, sha = profile.path, profile.doc_id, profile.source_sha256
    backend = profile.recommended_backend
    notes: List[str] = []
    text = ""
    units: List[SourceUnit] = []

    want_docling = backend in (BACKEND_DOCLING_OCR, BACKEND_DOCLING_TABLE)
    if want_docling and not allow_docling:
        notes.append("docling disabled by caller; using legacy")
        backend = BACKEND_LEGACY
        want_docling = False
    if (
        want_docling
        and page_range is None
        and profile.page_count > docling_page_cap
        and backend == BACKEND_DOCLING_TABLE
    ):
        # Doc exceeds the bound: fall back to legacy extraction for the FULL
        # document rather than truncating to a window and still running
        # Docling on it. Slicing-then-Docling silently drops every page past
        # the cap (e.g. a 144-page report would only ever ingest its front
        # matter) with no signal reaching the caller. Legacy loses table
        # cell/row/column structure but covers every page, which is the
        # documented contract for this cap.
        msg = (
            f"{doc_id}: {profile.page_count} pages > docling_page_cap="
            f"{docling_page_cap}; falling back to legacy extraction for full "
            f"page coverage (docling table-structure skipped to bound "
            f"runtime/memory on this document)"
        )
        logger.warning(f"[ingest] {msg}")
        notes.append(msg)
        backend = BACKEND_LEGACY
        want_docling = False

    if want_docling:
        try:
            from rag_gt.ingestion.docling_pdf import extract_pdf_docling_with_layout

            do_ocr = backend == BACKEND_DOCLING_OCR
            do_tables = backend == BACKEND_DOCLING_TABLE
            # Docling's with_layout helper has no page_range arg; for bounded
            # runs we extract the window via a temporary single-doc slice.
            src_path = path
            if page_range is not None:
                src_path = _slice_pdf(path, page_range)
            text, units = extract_pdf_docling_with_layout(
                src_path,
                doc_id,
                source_sha1=sha,
                do_ocr=do_ocr,
                do_table_structure=do_tables,
                min_text_chars=0,
                min_text_file_ratio=0.0,
            )
            if not units:
                raise RuntimeError("docling returned 0 units")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[ingest] docling failed for {doc_id}: {type(e).__name__}: {e}")
            notes.append(f"docling failed ({type(e).__name__}); fell back to legacy")
            backend = BACKEND_LEGACY
            text, units = _legacy(path, doc_id, sha, page_range)
    else:
        text, units = _legacy(path, doc_id, sha, page_range)

    f, b = _tag_matter(units, profile)
    pages = len({u.page_no for u in units if u.page_no is not None})
    return IngestResult(
        doc_id=doc_id,
        backend_used=backend,
        text=text,
        units=units,
        char_count=len(text),
        n_units=len(units),
        pages_covered=pages,
        front_matter_units=f,
        back_matter_units=b,
        notes=notes,
    )


def _slice_pdf(path: str, page_range: Tuple[int, int]) -> str:
    """Write a PDF slice for ``page_range`` (1-based inclusive) to a stable cache
    path. A stable path under the repo avoids the Windows temp-file lock that
    Docling's backend hits when it tries to manage an OS temp file.
    """
    import hashlib
    from pathlib import Path

    import fitz

    start, end = page_range
    # Anchor to the repo's data dir rather than a bare relative path: a bare
    # "data/cache/..." lands wherever the process happened to be launched from,
    # so the same slice got re-cut per working directory and stray data/ trees
    # appeared next to whatever script was run. data_dir() also honours
    # RAG_GT_DATA_DIR, matching the rest of the codebase.
    from rag_gt.core.config import data_dir

    cache_dir = data_dir() / "cache" / "allpdf_slices"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Key on file identity, not just its path: the same filename can hold a
    # different document later (corrected re-export, overwritten temp name).
    # Without size+mtime the cache would serve the previous document's pages
    # and ground truth would be built from a file no longer on disk.
    st = Path(path).stat()
    key = hashlib.sha1(
        f"{path}:{start}:{end}:{st.st_size}:{st.st_mtime_ns}".encode()
    ).hexdigest()[:16]
    out_path = cache_dir / f"slice_{key}.pdf"
    if out_path.exists():
        return str(out_path)
    src = fitz.open(path)
    out = fitz.open()
    out.insert_pdf(src, from_page=max(0, start - 1), to_page=min(src.page_count, end) - 1)
    out.save(str(out_path))
    out.close()
    src.close()
    return str(out_path)
