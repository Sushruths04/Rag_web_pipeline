"""Optional Docling PDF extraction.

Docling preserves document structure better than raw PDF text extraction for
many technical PDFs. The pipeline treats it as an optional front-end: callers
can try it first and fall back to the legacy pdfplumber/PyMuPDF extractor when
Docling is unavailable or fails on a file.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from rag_gt.core.types import SourceBBox, SourceUnit
from rag_gt.ingestion.cleaning import clean_text
from rag_gt.ingestion.pdf import extract_pdf_page_range


def extract_pdf_docling(
    path: str,
    export_format: str = "markdown",
    do_ocr: bool = False,
    do_table_structure: bool = False,
    batch_size: int = 1,
    min_text_chars: int = 1000,
    min_text_file_ratio: float = 0.05,
    page_range_size: int = 20,
    fallback_to_legacy_pages: bool = False,
) -> str:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except Exception as e:  # pragma: no cover - depends on optional install
        raise RuntimeError(f"Docling is not available: {type(e).__name__}: {e}") from e

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = bool(do_ocr)
    pipeline_options.do_table_structure = bool(do_table_structure)
    pipeline_options.force_backend_text = not bool(do_ocr)
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False
    pipeline_options.generate_parsed_pages = False
    pipeline_options.ocr_batch_size = max(1, int(batch_size))
    pipeline_options.layout_batch_size = max(1, int(batch_size))
    pipeline_options.table_batch_size = max(1, int(batch_size))
    pipeline_options.accelerator_options.num_threads = 1
    source = Path(path)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    ranges = _page_ranges(source, page_range_size)
    parts: list[str] = []
    docling_chars = 0
    legacy_fallback_ranges: list[Tuple[int, int]] = []

    for page_range in ranges:
        try:
            text = _convert_range(
                converter=converter,
                source=source,
                export_format=export_format,
                page_range=page_range,
            )
            docling_chars += len(text.strip())
        except Exception as e:
            if not fallback_to_legacy_pages:
                raise RuntimeError(
                    f"Docling failed for pages {page_range[0]}-{page_range[1]} "
                    f"of {source}: {type(e).__name__}: {e}"
                ) from e
            logger.warning(
                f"[Docling] pages {page_range[0]}-{page_range[1]} failed for "
                f"{source}; using legacy extraction for those pages: "
                f"{type(e).__name__}: {e}"
            )
            text = extract_pdf_page_range(str(source), page_range=page_range)
            legacy_fallback_ranges.append(page_range)
        parts.append(text)

    text = "\n\n".join(p for p in parts if p and p.strip())

    if legacy_fallback_ranges:
        logger.warning(
            f"[Docling] used legacy fallback for {len(legacy_fallback_ranges)} "
            f"page range(s) in {source}"
        )
    logger.info(
        f"[Docling] extracted {len(text.strip())} chars from {source} "
        f"({docling_chars} chars from Docling)"
    )

    text_len = len((text or "").strip())
    min_by_size = int(source.stat().st_size * float(min_text_file_ratio))
    min_required = max(int(min_text_chars), min_by_size)
    if text_len < min_required:
        raise RuntimeError(
            f"Docling produced too little text: {text_len} chars "
            f"(required >= {min_required})"
        )
    return text


def extract_pdf_docling_with_layout(
    path: str,
    doc_id: str,
    source_sha1: str = "",
    do_ocr: bool = False,
    do_table_structure: bool = False,
    batch_size: int = 1,
    min_text_chars: int = 1000,
    min_text_file_ratio: float = 0.05,
    page_range_size: int = 20,
    fallback_to_legacy_pages: bool = False,
) -> tuple[str, List[SourceUnit]]:
    """Extract canonical text plus Docling page/bbox source units."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except Exception as e:  # pragma: no cover - depends on optional install
        raise RuntimeError(f"Docling is not available: {type(e).__name__}: {e}") from e

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = bool(do_ocr)
    pipeline_options.do_table_structure = bool(do_table_structure)
    pipeline_options.force_backend_text = not bool(do_ocr)
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False
    pipeline_options.generate_parsed_pages = False
    pipeline_options.ocr_batch_size = max(1, int(batch_size))
    pipeline_options.layout_batch_size = max(1, int(batch_size))
    pipeline_options.table_batch_size = max(1, int(batch_size))
    pipeline_options.accelerator_options.num_threads = 1

    source = Path(path)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    parts: List[str] = []
    units: List[SourceUnit] = []
    cursor = 0
    docling_chars = 0
    legacy_fallback_ranges: list[Tuple[int, int]] = []

    for page_range in _page_ranges(source, page_range_size):
        try:
            result = converter.convert(
                source, raises_on_error=True, page_range=page_range
            )
            page_text, page_units = _docling_units_to_text(
                document=result.document,
                doc_id=doc_id,
                source_path=str(source),
                source_sha1=source_sha1,
                cursor=cursor,
            )
            def _retry_one_page(page_no: int) -> List[SourceUnit]:
                """Re-convert a single dropped page on its own.

                The batch-level bad_alloc does not recur at this size, so this
                usually returns the page's real (table-bearing) items.
                """
                rr = converter.convert(
                    source, raises_on_error=True, page_range=(page_no, page_no)
                )
                _, u = _docling_units_to_text(
                    document=rr.document,
                    doc_id=doc_id,
                    source_path=str(source),
                    source_sha1=source_sha1,
                    cursor=0,
                )
                return u

            # convert() can succeed while silently dropping pages; repair them.
            page_text, page_units = _repair_dropped_pages(
                page_text,
                page_units,
                source=source,
                doc_id=doc_id,
                source_sha1=source_sha1,
                page_range=page_range,
                cursor=cursor,
                docling_retry=_retry_one_page,
            )
            if page_text:
                if parts:
                    parts.append("\n\n")
                    cursor += 2
                    page_units = _shift_units(page_units, 2)
                parts.append(page_text)
                cursor += len(page_text)
                units.extend(page_units)
                docling_chars += len(page_text.strip())
        except Exception as e:
            if not fallback_to_legacy_pages:
                raise RuntimeError(
                    f"Docling layout extraction failed for pages "
                    f"{page_range[0]}-{page_range[1]} of {source}: "
                    f"{type(e).__name__}: {e}"
                ) from e
            from rag_gt.ingestion.pdf import extract_pdf_with_layout

            logger.warning(
                f"[Docling] layout pages {page_range[0]}-{page_range[1]} failed "
                f"for {source}; using PyMuPDF layout fallback: {type(e).__name__}: {e}"
            )
            page_text, page_units = extract_pdf_with_layout(
                str(source),
                doc_id=doc_id,
                source_sha1=source_sha1,
                page_range=page_range,
            )
            if page_text:
                if parts:
                    parts.append("\n\n")
                    cursor += 2
                shift = cursor
                page_units = _shift_units(page_units, shift)
                parts.append(page_text)
                cursor += len(page_text)
                units.extend(page_units)
            legacy_fallback_ranges.append(page_range)

    text = "".join(parts).strip()
    if legacy_fallback_ranges:
        logger.warning(
            f"[Docling] used PyMuPDF layout fallback for "
            f"{len(legacy_fallback_ranges)} page range(s) in {source}"
        )
    logger.info(
        f"[Docling] extracted {len(text)} chars with {len(units)} source units "
        f"from {source} ({docling_chars} chars from Docling)"
    )
    _validate_min_text(source, text, min_text_chars, min_text_file_ratio)
    return text, units


def _effective_pages(units: List[SourceUnit]) -> List[int]:
    """Page number per unit, carrying the last seen page forward.

    Docling emits some items without provenance (page_no is None). Those
    belong with whatever preceded them in reading order, so they inherit the
    previous unit's page rather than sorting to the front.
    """
    out: List[int] = []
    last = 0
    for u in units:
        if u.page_no is not None:
            last = u.page_no
        out.append(last)
    return out


def _reserialize(units: List[SourceUnit], cursor: int) -> tuple[str, List[SourceUnit]]:
    """Join an ordered unit list into text, reassigning char offsets from cursor.

    Mirrors the "\\n\\n" joining used when Docling items are first serialized,
    so a repaired range is indistinguishable in shape from an intact one.
    """
    parts: List[str] = []
    out: List[SourceUnit] = []
    local = 0
    for u in units:
        if parts:
            parts.append("\n\n")
            local += 2
        start = cursor + local
        parts.append(u.text)
        local += len(u.text)
        out.append(
            SourceUnit(
                doc_id=u.doc_id,
                char_start=start,
                char_end=cursor + local,
                text=u.text,
                page_no=u.page_no,
                block_id=u.block_id,
                paragraph_id=u.paragraph_id,
                bboxes=u.bboxes,
                source_path=u.source_path,
                source_sha1=u.source_sha1,
                extractor=u.extractor,
            )
        )
    return "".join(parts), out


def _repair_dropped_pages(
    page_text: str,
    page_units: List[SourceUnit],
    *,
    source: Path,
    doc_id: str,
    source_sha1: str,
    page_range: Tuple[int, int],
    cursor: int,
    docling_retry=None,
) -> tuple[str, List[SourceUnit]]:
    """Fill in pages Docling dropped without raising.

    Docling swallows some native-layer failures (notably ``std::bad_alloc``
    during preprocessing) and returns a *partial* document: the convert call
    succeeds, so the caller's ``except`` never fires, and the affected pages
    disappear with no signal. Observed on DIN EN ISO 13919-1 (pages 18-20,
    non-deterministically 17-20) where the dropped pages held the imperfection
    classification tables -- i.e. the document's entire technical substance.

    Any page in the range that produced no unit is recovered and spliced back
    in page order. Recovery is tried Docling-first: the bad_alloc is memory
    pressure from the whole batch, not a property of the pages -- pages 19 and
    20 of that document convert cleanly on their own, each yielding a
    TableItem. Retrying them narrowly therefore keeps the table structure the
    docling_table backend exists to produce; only if that also comes back
    empty do we settle for PyMuPDF's flat text. Genuinely blank pages yield
    nothing either way and are self-correcting.
    """
    start, end = page_range
    covered = {u.page_no for u in page_units if u.page_no is not None}
    missing = [p for p in range(start, end + 1) if p not in covered]
    if not missing:
        return page_text, page_units

    from rag_gt.ingestion.pdf import extract_pdf_with_layout

    recovered: List[SourceUnit] = []
    via_docling: List[int] = []
    via_pymupdf: List[int] = []
    for p in missing:
        if docling_retry is not None:
            try:
                retried = docling_retry(p) or []
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"[Docling] narrow retry of page {p} failed: "
                    f"{type(e).__name__}: {e}; falling back to PyMuPDF"
                )
                retried = []
            # Only accept units the retry actually attributed to this page: a
            # converter that ignores page_range would otherwise splice another
            # page's content in under page p's identity.
            retried = [u for u in retried if u.page_no in (p, None)]
            for u in retried:
                u.page_no = p
            if retried:
                recovered.extend(retried)
                via_docling.append(p)
                continue
        try:
            _, u = extract_pdf_with_layout(
                str(source), doc_id=doc_id, source_sha1=source_sha1, page_range=(p, p)
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[Docling] page {p} repair failed: {type(e).__name__}: {e}")
            continue
        if u:
            recovered.extend(u)
            via_pymupdf.append(p)

    if not recovered:
        return page_text, page_units

    how = []
    if via_docling:
        how.append(f"{via_docling} via narrow Docling retry (structure kept)")
    if via_pymupdf:
        how.append(f"{via_pymupdf} via PyMuPDF (flat text)")
    logger.warning(
        f"[Docling] {source.name}: pages {missing} produced no units "
        f"(silent Docling drop); recovered {len(recovered)} unit(s): "
        + "; ".join(how)
    )
    merged = page_units + recovered
    keys = _effective_pages(page_units) + [
        u.page_no if u.page_no is not None else 0 for u in recovered
    ]
    order = sorted(range(len(merged)), key=lambda i: keys[i])
    return _reserialize([merged[i] for i in order], cursor)


def _page_ranges(source: Path, page_range_size: int) -> list[Tuple[int, int]]:
    import fitz

    with fitz.open(source) as pdf:
        page_count = len(pdf)
    if page_count <= 0:
        return []
    size = max(1, int(page_range_size or page_count))
    return [
        (start, min(start + size - 1, page_count))
        for start in range(1, page_count + 1, size)
    ]


def _convert_range(
    converter,
    source: Path,
    export_format: str,
    page_range: Tuple[int, int],
) -> str:
    result = converter.convert(source, raises_on_error=True, page_range=page_range)
    document = result.document
    if export_format == "text":
        return document.export_to_text()
    return document.export_to_markdown()


def _item_text(item, document) -> str:
    """Text for one Docling item, serializing tables instead of dropping them.

    A ``TableItem``'s ``.text`` is ``None``, so a plain ``getattr(item,
    "text")`` silently discarded every table -- on the very backend chosen
    *because* the document is table-dense, with ``do_table_structure=True``
    already paying to extract that structure. ``export_to_markdown()`` keeps
    the header row and cell alignment, which is what lets a downstream stage
    read a row against its header instead of seeing loose cells.
    """
    raw = getattr(item, "text", None)
    if raw:
        return str(raw)
    exporter = getattr(item, "export_to_markdown", None)
    if exporter is None:
        return ""
    try:
        try:
            return str(exporter(document) or "")
        except TypeError:
            # Older Docling signatures take no document argument.
            return str(exporter() or "")
    except Exception as e:  # noqa: BLE001
        ref = getattr(item, "self_ref", "?")
        logger.warning(
            f"[Docling] could not serialize {type(item).__name__} {ref}: "
            f"{type(e).__name__}: {e}; skipping it"
        )
        return ""


def _docling_units_to_text(
    document,
    doc_id: str,
    source_path: str,
    source_sha1: str,
    cursor: int,
) -> tuple[str, List[SourceUnit]]:
    parts: List[str] = []
    units: List[SourceUnit] = []
    local_cursor = 0
    paragraph_idx = 0
    for item, _level in document.iterate_items():
        cleaned = clean_text(_item_text(item, document))
        if not cleaned:
            continue
        if parts:
            parts.append("\n\n")
            local_cursor += 2
        start = cursor + local_cursor
        parts.append(cleaned)
        local_cursor += len(cleaned)
        paragraph_idx += 1
        bboxes: List[SourceBBox] = []
        pages: List[int] = []
        for prov in getattr(item, "prov", []) or []:
            page_no = int(getattr(prov, "page_no", 0) or 0)
            if page_no:
                pages.append(page_no)
            bbox = getattr(prov, "bbox", None)
            if bbox is not None:
                origin = getattr(getattr(bbox, "coord_origin", ""), "value", None)
                bboxes.append(
                    SourceBBox(
                        page_no=page_no,
                        l=float(getattr(bbox, "l", 0.0) or 0.0),
                        t=float(getattr(bbox, "t", 0.0) or 0.0),
                        r=float(getattr(bbox, "r", 0.0) or 0.0),
                        b=float(getattr(bbox, "b", 0.0) or 0.0),
                        coord_origin=str(origin or "BOTTOMLEFT"),
                    )
                )
        block_id = str(getattr(item, "self_ref", "") or f"p{paragraph_idx}")
        page_no = min(pages) if pages else None
        units.append(
            SourceUnit(
                doc_id=doc_id,
                char_start=start,
                char_end=cursor + local_cursor,
                text=cleaned,
                page_no=page_no,
                block_id=block_id,
                paragraph_id=block_id or f"para_{paragraph_idx}",
                bboxes=bboxes,
                source_path=source_path,
                source_sha1=source_sha1,
                extractor="docling",
            )
        )
    return "".join(parts), units


def _shift_units(units: List[SourceUnit], delta: int) -> List[SourceUnit]:
    if not delta:
        return units
    shifted: List[SourceUnit] = []
    for u in units:
        shifted.append(
            SourceUnit(
                doc_id=u.doc_id,
                char_start=u.char_start + delta,
                char_end=u.char_end + delta,
                text=u.text,
                page_no=u.page_no,
                block_id=u.block_id,
                paragraph_id=u.paragraph_id,
                bboxes=u.bboxes,
                source_path=u.source_path,
                source_sha1=u.source_sha1,
                extractor=u.extractor,
            )
        )
    return shifted


def _validate_min_text(
    source: Path,
    text: str,
    min_text_chars: int,
    min_text_file_ratio: float,
) -> None:
    text_len = len((text or "").strip())
    min_by_size = int(source.stat().st_size * float(min_text_file_ratio))
    min_required = max(int(min_text_chars), min_by_size)
    if text_len < min_required:
        raise RuntimeError(
            f"Docling produced too little text: {text_len} chars "
            f"(required >= {min_required})"
        )
