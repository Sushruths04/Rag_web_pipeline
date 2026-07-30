"""Stage 0 — PDF Preflight / Router.

Profiles a PDF without committing to an extractor, so the pipeline can choose
the right ingestion + chunking strategy per document. Replaces the hardcoded
per-filename `doc_page_excludes` with automatic front/back-matter detection.

Pure-read, dependency-light (PyMuPDF + pdfplumber, both already required).
Expensive checks are sampled (evenly spaced pages) so it stays fast on
300-page books.
"""

from __future__ import annotations

import hashlib
import re
import statistics
from dataclasses import asdict, dataclass, field
from typing import List, Optional

_FRONT_MATTER_CUES = re.compile(
    r"\b(table of contents|copyright|all rights reserved|isbn|"
    r"acknowledg(e?ments)?|preface|foreword|about the author)\b",
    re.I,
)
_BACK_MATTER_CUES = re.compile(
    r"\b(references|bibliography|works cited|index|appendix|glossary)\b", re.I
)
_STANDARD_CUES = re.compile(
    r"\b(shall|clause|conformance|normative|this standard|specification)\b", re.I
)
_FINANCIAL_CUES = re.compile(
    r"\b(balance sheet|cash flow|fiscal year|net income|shareholders|"
    r"consolidated|total assets|revenues?)\b",
    re.I,
)

# Recommended backend tokens (consumed by Stage 1 ingestion router).
BACKEND_LEGACY = "legacy"          # pdfplumber/PyMuPDF text path
BACKEND_DOCLING_OCR = "docling_ocr"      # scanned/image PDFs
BACKEND_DOCLING_TABLE = "docling_table"  # table-dense digital PDFs


@dataclass
class DocProfile:
    doc_id: str
    path: str
    source_sha256: str
    page_count: int
    sampled_pages: int
    digital_text_ratio: float       # fraction of sampled pages with real text
    scanned: bool
    mixed: bool
    avg_chars_per_page: float
    figure_density: float           # mean embedded images per sampled page
    table_density: float            # fraction of sampled pages with >=1 table
    multi_column: bool
    reading_order_risk: str         # low | medium | high
    front_matter_pages: List[int] = field(default_factory=list)
    back_matter_pages: List[int] = field(default_factory=list)
    doc_type_guess: str = "UNKNOWN"
    recommended_backend: str = BACKEND_LEGACY
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sample_indices(page_count: int, k: int) -> List[int]:
    if page_count <= k:
        return list(range(page_count))
    step = page_count / k
    return sorted({int(i * step) for i in range(k)})


def _looks_multi_column(blocks: list) -> bool:
    """Bimodal block left-edge distribution => probable 2-column layout."""
    xs = [b[0] for b in blocks if len(b) >= 5 and (b[4] or "").strip()]
    if len(xs) < 6:
        return False
    xs_sorted = sorted(xs)
    mid = (min(xs) + max(xs)) / 2
    left = [x for x in xs_sorted if x < mid]
    right = [x for x in xs_sorted if x >= mid]
    # Two well-populated clusters separated by a gap.
    return len(left) >= 3 and len(right) >= 3 and (min(right) - max(left)) > 40


def _table_count_pdfplumber(path: str, indices: List[int]) -> Optional[int]:
    try:
        import pdfplumber
    except Exception:  # noqa: BLE001
        return None
    pages_with_tables = 0
    try:
        with pdfplumber.open(path) as pdf:
            n = len(pdf.pages)
            for i in indices:
                if i >= n:
                    continue
                try:
                    if pdf.pages[i].find_tables():
                        pages_with_tables += 1
                except Exception:  # noqa: BLE001
                    continue
    except Exception:  # noqa: BLE001
        return None
    return pages_with_tables


def profile_pdf(path: str, doc_id: str, sample_k: int = 24) -> DocProfile:
    import fitz

    notes: List[str] = []
    with fitz.open(path) as doc:
        page_count = doc.page_count
        idx = _sample_indices(page_count, sample_k)
        digital = 0
        char_counts: List[int] = []
        image_counts: List[int] = []
        multi_col_votes = 0
        front: List[int] = []
        back: List[int] = []

        for i in idx:
            page = doc[i]
            try:
                text = page.get_text("text") or ""
            except Exception:  # noqa: BLE001
                text = ""
            clen = len(text.strip())
            char_counts.append(clen)
            if clen >= 100:
                digital += 1
            try:
                image_counts.append(len(page.get_images(full=True)))
            except Exception:  # noqa: BLE001
                image_counts.append(0)
            try:
                blocks = page.get_text("blocks")
                if _looks_multi_column(blocks):
                    multi_col_votes += 1
            except Exception:  # noqa: BLE001
                pass

        # Front/back matter scan: first 15 and last 15 pages.
        for i in range(min(15, page_count)):
            t = doc[i].get_text("text") or ""
            if _FRONT_MATTER_CUES.search(t):
                front.append(i + 1)
        for i in range(max(0, page_count - 15), page_count):
            t = doc[i].get_text("text") or ""
            if _BACK_MATTER_CUES.search(t):
                back.append(i + 1)

        # Aggregate first-page-ish text for doc-type cues.
        sample_text = " ".join(
            (doc[i].get_text("text") or "")[:2000] for i in idx[: min(8, len(idx))]
        )

    sampled = len(idx) or 1
    digital_ratio = digital / sampled
    avg_chars = statistics.mean(char_counts) if char_counts else 0.0
    fig_density = statistics.mean(image_counts) if image_counts else 0.0
    multi_col = multi_col_votes >= max(2, sampled // 4)

    tbl_pages = _table_count_pdfplumber(path, idx)
    if tbl_pages is None:
        table_density = 0.0
        notes.append("table detection unavailable")
    else:
        table_density = tbl_pages / sampled

    scanned = digital_ratio < 0.2
    mixed = 0.2 <= digital_ratio < 0.7
    reading_order_risk = "high" if multi_col else ("medium" if mixed else "low")

    # Doc-type heuristic mapped to existing DocType enum vocabulary.
    if scanned:
        doc_type = "UNKNOWN"
    elif _STANDARD_CUES.search(sample_text) and table_density < 0.5:
        doc_type = "ISO_STANDARD"
    elif _FINANCIAL_CUES.search(sample_text) or table_density >= 0.5:
        doc_type = "REPORT"
    elif page_count >= 120 and avg_chars >= 800:
        doc_type = "TEXTBOOK"
    elif avg_chars >= 400:
        doc_type = "NARRATIVE"
    else:
        doc_type = "UNKNOWN"

    if scanned:
        backend = BACKEND_DOCLING_OCR
    elif table_density >= 0.4:
        backend = BACKEND_DOCLING_TABLE
    else:
        backend = BACKEND_LEGACY

    return DocProfile(
        doc_id=doc_id,
        path=path,
        source_sha256=_sha256(path),
        page_count=page_count,
        sampled_pages=sampled,
        digital_text_ratio=round(digital_ratio, 3),
        scanned=scanned,
        mixed=mixed,
        avg_chars_per_page=round(avg_chars, 1),
        figure_density=round(fig_density, 2),
        table_density=round(table_density, 3),
        multi_column=multi_col,
        reading_order_risk=reading_order_risk,
        front_matter_pages=front,
        back_matter_pages=back,
        doc_type_guess=doc_type,
        recommended_backend=backend,
        notes=notes,
    )
