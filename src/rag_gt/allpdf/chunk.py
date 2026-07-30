"""Stage 2 — Agentic chunking router.

Different documents need different chunking. A scanned page, a financial table,
an ISO clause, and a textbook paragraph do not chunk well under one strategy.
This router picks the strategy AND the size/overlap budget per document from its
Stage-0 profile (the "agentic" decision), then either delegates to the proven
text chunkers or uses a structure-preserving unit-packing chunker.

Strategies
----------
- clause      : ISO_STANDARD — split on numbered clauses (tight chunks)
- heading     : TEXTBOOK     — split on headings, sentence-packed
- recursive   : NARRATIVE/paper — recursive character splitter
- table_aware : REPORT/financial — pack source units, never split a table block
- ocr_block   : scanned — pack OCR blocks (layout is weak, keep blocks intact)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from rag_gt.allpdf.ingest import IngestResult
from rag_gt.allpdf.preflight import DocProfile
from rag_gt.chunking.strategies import chunk_document
from rag_gt.core.types import Document, SourceUnit

# strategy -> (route_doc_type_for_text_chunkers, chunk_size, overlap)
_PARAMS = {
    "clause": ("ISO_STANDARD", 400, 40),
    "heading": ("TEXTBOOK", 700, 100),
    "recursive": ("NARRATIVE", 512, 80),
    "table_aware": ("REPORT", 900, 0),
    "ocr_block": ("UNKNOWN", 600, 40),
}


@dataclass
class ChunkResult:
    doc_id: str
    strategy: str
    chunk_size: int
    overlap: int
    chunks: List[dict]

    def summary(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "strategy": self.strategy,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "n_chunks": len(self.chunks),
        }


def select_strategy(profile: DocProfile) -> str:
    if profile.scanned:
        return "ocr_block"
    if profile.table_density >= 0.4 or profile.doc_type_guess == "REPORT":
        return "table_aware"
    if profile.doc_type_guess == "ISO_STANDARD":
        return "clause"
    if profile.doc_type_guess == "TEXTBOOK":
        return "heading"
    return "recursive"


def _pack_units(
    doc_id: str, units: List[SourceUnit], chunk_size: int
) -> List[dict]:
    """Greedily pack whole source units into chunks without splitting any unit.

    A docling table or an OCR block is a single unit, so this keeps tables and
    scanned blocks intact. Each chunk records page span and bboxes from its
    constituent units, preserving provenance.
    """
    chunks: List[dict] = []
    cur: List[SourceUnit] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if not cur:
            return
        text = "\n".join(u.text for u in cur).strip()
        if not text:
            cur, cur_len = [], 0
            return
        pages = [u.page_no for u in cur if u.page_no is not None]
        bboxes = [bb.to_dict() for u in cur for bb in u.bboxes]
        idx = len(chunks)
        chunks.append({
            "chunk_id": f"{doc_id}_c{idx:06d}",
            "doc_id": doc_id,
            "text": text,
            "char_start": cur[0].char_start,
            "char_end": cur[-1].char_end,
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
            "n_source_units": len(cur),
            "bboxes": bboxes,
        })
        cur, cur_len = [], 0

    for u in units:
        ulen = len(u.text or "")
        # A single oversized unit (e.g. a big table) becomes its own chunk.
        if ulen >= chunk_size and not cur:
            cur = [u]
            flush()
            continue
        if cur_len + ulen > chunk_size and cur:
            flush()
        cur.append(u)
        cur_len += ulen
    flush()
    return chunks


def agentic_chunk(profile: DocProfile, ingest: IngestResult) -> ChunkResult:
    strategy = select_strategy(profile)
    route_type, size, overlap = _PARAMS[strategy]

    if strategy in ("table_aware", "ocr_block"):
        chunks = _pack_units(profile.doc_id, ingest.units, size)
    else:
        doc = Document(
            doc_id=profile.doc_id,
            text=ingest.text,
            doc_type=route_type,  # routes the existing chunkers
            source_path=profile.path,
            source_sha1=profile.source_sha256,
            source_units=ingest.units,
        )
        chunks = chunk_document(doc, {"doc_type": route_type}, size, overlap)

    return ChunkResult(
        doc_id=profile.doc_id,
        strategy=strategy,
        chunk_size=size,
        overlap=overlap,
        chunks=chunks,
    )
