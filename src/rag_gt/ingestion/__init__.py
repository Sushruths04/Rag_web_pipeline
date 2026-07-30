"""Ingestion layer: PDF (primary) + DOCX (optional) + text cleaning."""

from __future__ import annotations

import os
import re
import hashlib

from loguru import logger

from rag_gt.core.config import load_config
from rag_gt.core.types import DocType, Document
from rag_gt.ingestion.cleaning import clean_text
from rag_gt.ingestion.pdf import extract_pdf, extract_pdf_with_layout

_VERSION_TAIL_RE = re.compile(r"_v(\d+(?:\.\d+)*)$")


def ingest_document(path: str, doc_type: DocType = "UNKNOWN") -> Document:
    ext = os.path.splitext(path)[1].lower()
    doc_id = os.path.splitext(os.path.basename(path))[0]
    source_sha1 = _file_sha1(path)
    source_backend = ""
    source_units = []

    if ext == ".pdf":
        cfg = load_config()
        ingestion_cfg = cfg.get("ingestion", {})
        backend = str(ingestion_cfg.get("pdf_backend", "legacy")).lower()
        docling_do_ocr = bool(ingestion_cfg.get("docling_do_ocr", False))
        docling_do_table_structure = bool(
            ingestion_cfg.get("docling_do_table_structure", False)
        )
        docling_batch_size = int(ingestion_cfg.get("docling_batch_size", 1))
        docling_page_range_size = int(
            ingestion_cfg.get("docling_page_range_size", 20)
        )
        docling_min_text_chars = int(
            ingestion_cfg.get("docling_min_text_chars", 1000)
        )
        docling_min_text_file_ratio = float(
            ingestion_cfg.get("docling_min_text_file_ratio", 0.05)
        )

        if backend in {"docling", "auto"}:
            try:
                from rag_gt.ingestion.docling_pdf import extract_pdf_docling_with_layout

                text, source_units = extract_pdf_docling_with_layout(
                    path,
                    doc_id=doc_id,
                    source_sha1=source_sha1,
                    do_ocr=docling_do_ocr,
                    do_table_structure=docling_do_table_structure,
                    batch_size=docling_batch_size,
                    page_range_size=docling_page_range_size,
                    min_text_chars=docling_min_text_chars,
                    min_text_file_ratio=docling_min_text_file_ratio,
                    fallback_to_legacy_pages=(backend == "auto"),
                )
                source_backend = "docling"
            except Exception as e:
                if backend == "docling":
                    raise
                logger.warning(
                    f"[Docling] falling back to legacy PDF extraction for {path}: "
                    f"{type(e).__name__}: {e}"
                )
                text, source_units = extract_pdf_with_layout(
                    path, doc_id=doc_id, source_sha1=source_sha1
                )
                source_backend = "pymupdf"
        else:
            text, source_units = extract_pdf_with_layout(
                path, doc_id=doc_id, source_sha1=source_sha1
            )
            source_backend = "pymupdf"
    elif ext in (".docx", ".doc"):
        cfg = load_config()
        if not cfg.get("ingestion", {}).get("enable_docx", False):
            raise ValueError(
                "DOCX ingestion is disabled. "
                "Set ingestion.enable_docx=true in config.yaml to enable."
            )
        from rag_gt.ingestion.docx import extract_docx

        text = extract_docx(path)
    else:
        raise ValueError(
            f"Unsupported format '{ext}'. Supported: .pdf (and .docx, opt-in)."
        )

    if ext == ".pdf":
        # Source-aware PDF extractors already return cleaned canonical text;
        # re-cleaning here would invalidate char offsets and bbox mappings.
        pass
    else:
        text = clean_text(text)
    m = _VERSION_TAIL_RE.search(doc_id)
    version = m.group(1) if m else ""

    return Document(
        doc_id=doc_id,
        text=text,
        version=version,
        doc_type=doc_type,
        source_path=os.path.abspath(path) if ext == ".pdf" else "",
        source_sha1=source_sha1 if ext == ".pdf" else "",
        source_backend=source_backend,
        source_units=source_units,
    )


def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = [
    "clean_text",
    "extract_docx",
    "extract_pdf",
    "extract_pdf_with_layout",
    "extract_pdf_docling",
    "extract_pdf_docling_with_layout",
    "ingest_document",
]


def __getattr__(name: str):
    """Lazy import for extract_docx so the package import doesn't fail when
    python-docx isn't installed (DOCX is opt-in)."""
    if name == "extract_docx":
        from rag_gt.ingestion.docx import extract_docx as _extract_docx

        return _extract_docx
    if name == "extract_pdf_docling":
        from rag_gt.ingestion.docling_pdf import extract_pdf_docling as _extract_pdf_docling

        return _extract_pdf_docling
    if name == "extract_pdf_docling_with_layout":
        from rag_gt.ingestion.docling_pdf import (
            extract_pdf_docling_with_layout as _extract_pdf_docling_with_layout,
        )

        return _extract_pdf_docling_with_layout
    raise AttributeError(name)
