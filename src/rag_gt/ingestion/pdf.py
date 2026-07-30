"""PDF text extraction (pdfplumber primary, PyMuPDF fallback).

Strategy:
- Open pdfplumber once.
- For each page, try pdfplumber. If the result is short (<50 chars), fall back
  to a single shared PyMuPDF (fitz) document for the same page.
- If pdfplumber fails outright, fall back to fitz for the whole document.
- All file handles are closed via try/finally; legitimate short pages (covers,
  TOC, etc.) are preserved — no length-based filter at the join step.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from loguru import logger

from rag_gt.core.types import SourceBBox, SourceUnit
from rag_gt.ingestion.cleaning import clean_text


def _page_bounds(page_count: int, page_range: Optional[Tuple[int, int]]) -> tuple[int, int]:
    if page_range is None:
        return 0, page_count
    start, end = page_range
    start_idx = max(0, start - 1)
    end_idx = min(page_count, end)
    if start_idx >= end_idx:
        return 0, 0
    return start_idx, end_idx


def extract_pdf(path: str) -> str:
    return extract_pdf_page_range(path)


def extract_pdf_with_layout(
    path: str,
    doc_id: str,
    source_sha1: str = "",
    page_range: Optional[Tuple[int, int]] = None,
) -> tuple[str, List[SourceUnit]]:
    """Extract PDF text with page/block bbox provenance using PyMuPDF.

    This is the deterministic fallback source map. It is less structural than
    Docling but preserves page and bbox traceability instead of returning plain
    text with no source coordinates.
    """
    import fitz

    parts: List[str] = []
    units: List[SourceUnit] = []
    cursor = 0
    with fitz.open(path) as pdf:
        start_idx, end_idx = _page_bounds(len(pdf), page_range)
        for page_idx in range(start_idx, end_idx):
            page = pdf[page_idx]
            page_no = page_idx + 1
            blocks = page.get_text("blocks")
            for block_idx, block in enumerate(blocks):
                if len(block) < 5:
                    continue
                x0, y0, x1, y1, text = block[:5]
                cleaned = clean_text(str(text or ""))
                if not cleaned:
                    continue
                if parts:
                    parts.append("\n\n")
                    cursor += 2
                start = cursor
                parts.append(cleaned)
                cursor += len(cleaned)
                units.append(
                    SourceUnit(
                        doc_id=doc_id,
                        char_start=start,
                        char_end=cursor,
                        text=cleaned,
                        page_no=page_no,
                        block_id=f"p{page_no}_b{block_idx}",
                        paragraph_id=f"p{page_no}_b{block_idx}",
                        bboxes=[
                            SourceBBox(
                                page_no=page_no,
                                l=float(x0),
                                t=float(y0),
                                r=float(x1),
                                b=float(y1),
                                coord_origin="TOPLEFT",
                            )
                        ],
                        source_path=path,
                        source_sha1=source_sha1,
                        extractor="pymupdf",
                    )
                )
    return "".join(parts).strip(), units


def extract_pdf_page_range(
    path: str, page_range: Optional[Tuple[int, int]] = None
) -> str:
    import fitz
    import pdfplumber

    pages: List[str] = []
    fitz_doc = None
    try:
        try:
            pdf_ctx = pdfplumber.open(path)
        except Exception as e:
            logger.warning(f"[PDF] pdfplumber.open failed for {path}: {type(e).__name__}: {e}")
            pdf_ctx = None

        if pdf_ctx is not None:
            with pdf_ctx as pdf:
                start_idx, end_idx = _page_bounds(len(pdf.pages), page_range)
                for i in range(start_idx, end_idx):
                    page = pdf.pages[i]
                    try:
                        text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                    except Exception as e:
                        logger.warning(
                            f"[PDF] pdfplumber page {i} failed for {path}: {type(e).__name__}; falling back to fitz"
                        )
                        text = ""
                    if len(text.strip()) < 50:
                        if fitz_doc is None:
                            fitz_doc = fitz.open(path)
                        if i < len(fitz_doc):
                            text = fitz_doc[i].get_text("text")
                        else:
                            logger.warning(
                                f"[PDF] page index {i} out of range for fitz on {path} "
                                f"(fitz pages={len(fitz_doc)})"
                            )
                    pages.append(text)
        else:
            # pdfplumber failed at open time — full fitz fallback.
            fitz_doc = fitz.open(path)
            start_idx, end_idx = _page_bounds(len(fitz_doc), page_range)
            for i in range(start_idx, end_idx):
                pages.append(fitz_doc[i].get_text("text"))
    finally:
        if fitz_doc is not None:
            try:
                fitz_doc.close()
            except Exception:
                pass

    # Keep all pages; downstream chunking handles short pages. Joining with
    # double-newline preserves page boundaries for heading-aware chunkers.
    return "\n\n".join(pages)
