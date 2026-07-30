"""Source-span utilities for PDF-grounded facts and chunk mappings.

The canonical truth is a character/page/bbox range in ``Document.text``. Chunk
IDs are strategy-specific projections derived from those ranges.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from rag_gt.core.types import Document, SourceBBox, SourceUnit, Span


@dataclass(frozen=True)
class ChunkOverlap:
    chunk_id: str
    doc_id: str
    char_start: int
    char_end: int
    overlap_chars: int
    overlap_ratio: float
    pages: List[int] = field(default_factory=list)


def sha1_text(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def attach_source_units_to_range(
    doc: Document,
    char_start: int,
    char_end: int,
) -> dict:
    """Return source metadata covering a document character range."""
    units = units_for_range(doc.source_units, char_start, char_end)
    pages = sorted({u.page_no for u in units if u.page_no is not None})
    bboxes: List[SourceBBox] = []
    block_ids: List[str] = []
    paragraph_ids: List[str] = []
    for u in units:
        bboxes.extend(u.bboxes)
        if u.block_id:
            block_ids.append(u.block_id)
        if u.paragraph_id:
            paragraph_ids.append(u.paragraph_id)

    return {
        "page_start": pages[0] if pages else None,
        "page_end": pages[-1] if pages else None,
        "bboxes": bboxes,
        "block_ids": _dedupe(block_ids),
        "paragraph_ids": _dedupe(paragraph_ids),
        "source_path": doc.source_path,
        "source_sha1": doc.source_sha1,
        "extractor": doc.source_backend,
        "source_text_sha1": sha1_text(doc.text[char_start:char_end]),
    }


def units_for_range(
    units: Iterable[SourceUnit],
    char_start: int,
    char_end: int,
) -> List[SourceUnit]:
    if char_end <= char_start:
        return []
    return [
        u for u in units
        if u.char_end > char_start and u.char_start < char_end
    ]


def annotate_chunk_with_source(doc: Document, chunk: dict) -> dict:
    """Add source coverage metadata to a chunk dict in-place and return it."""
    cs = int(chunk.get("char_start", 0) or 0)
    ce = int(chunk.get("char_end", 0) or 0)
    units = units_for_range(doc.source_units, cs, ce)
    pages = sorted({u.page_no for u in units if u.page_no is not None})
    chunk["source_char_start"] = cs
    chunk["source_char_end"] = ce
    if pages:
        chunk["page_start"] = pages[0]
        chunk["page_end"] = pages[-1]
        chunk["pages"] = pages
    if doc.source_path:
        chunk["source_path"] = doc.source_path
    if doc.source_sha1:
        chunk["source_sha1"] = doc.source_sha1
    if doc.source_backend:
        chunk["extractor"] = doc.source_backend
    if units:
        chunk["source_units"] = [
            {
                "char_start": max(cs, u.char_start),
                "char_end": min(ce, u.char_end),
                "page_no": u.page_no,
                "block_id": u.block_id,
                "paragraph_id": u.paragraph_id,
                "bboxes": [b.to_dict() for b in u.bboxes],
            }
            for u in units
        ]
    return chunk


def span_to_chunk_overlaps(
    span: Span,
    chunks: Iterable[dict],
    *,
    min_overlap_ratio: float = 0.0,
) -> List[ChunkOverlap]:
    if span.char_start is None or span.char_end is None:
        return []
    out: List[ChunkOverlap] = []
    span_len = max(1, span.char_end - span.char_start)
    for c in chunks:
        if str(c.get("doc_id", "")) != span.doc_id:
            continue
        cs = _chunk_start(c)
        ce = _chunk_end(c)
        overlap = max(0, min(span.char_end, ce) - max(span.char_start, cs))
        if overlap <= 0:
            continue
        ratio = overlap / span_len
        if ratio < min_overlap_ratio:
            continue
        pages = [int(p) for p in c.get("pages", []) if str(p).isdigit()]
        out.append(
            ChunkOverlap(
                chunk_id=str(c.get("chunk_id", "")),
                doc_id=span.doc_id,
                char_start=cs,
                char_end=ce,
                overlap_chars=overlap,
                overlap_ratio=ratio,
                pages=pages,
            )
        )
    out.sort(key=lambda x: (-x.overlap_ratio, x.char_start, x.chunk_id))
    return out


def map_fact_to_chunks(
    spans: Iterable[Span],
    chunks: Iterable[dict],
    *,
    min_overlap_ratio: float = 0.0,
) -> List[str]:
    by_id: dict[str, ChunkOverlap] = {}
    chunk_list = list(chunks)
    for span in spans:
        for ov in span_to_chunk_overlaps(
            span, chunk_list, min_overlap_ratio=min_overlap_ratio
        ):
            current = by_id.get(ov.chunk_id)
            if current is None or ov.overlap_ratio > current.overlap_ratio:
                by_id[ov.chunk_id] = ov
    return [ov.chunk_id for ov in sorted(by_id.values(), key=lambda x: x.char_start)]


def _chunk_start(chunk: dict) -> int:
    return int(chunk.get("source_char_start", chunk.get("char_start", 0)) or 0)


def _chunk_end(chunk: dict) -> int:
    return int(chunk.get("source_char_end", chunk.get("char_end", 0)) or 0)


def _dedupe(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out
