"""Helpers for exporting external RAG chunks in the RAG-GT V11 contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Optional


def sha1_file(path: str | Path) -> str:
    h = hashlib.sha1()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_user_chunk_row(
    chunk: Mapping,
    *,
    doc_id: str,
    source_path: str | Path = "",
    source_sha1: str = "",
    default_chunk_id: str = "",
) -> dict:
    """Return one JSON-serializable chunk row for RAG-GT mapping.

    `chunk` may come from any user RAG system. The strongest mapping requires
    char offsets; page metadata enables a weaker text fallback.
    """

    chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or default_chunk_id)
    if not chunk_id:
        raise ValueError("chunk_id is required")
    text = str(chunk.get("text", "") or "")
    if not text:
        raise ValueError(f"text is required for chunk {chunk_id}")

    row = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": text,
        "source_path": str(source_path or chunk.get("source_path", "") or ""),
        "source_sha1": source_sha1 or str(chunk.get("source_sha1", "") or ""),
        "char_start": _optional_int(chunk.get("char_start", chunk.get("source_char_start"))),
        "char_end": _optional_int(chunk.get("char_end", chunk.get("source_char_end"))),
        "page_start": _optional_int(chunk.get("page_start")),
        "page_end": _optional_int(chunk.get("page_end", chunk.get("page_start"))),
        "paragraph_ids": list(chunk.get("paragraph_ids", []) or []),
        "block_ids": list(chunk.get("block_ids", []) or []),
        "bboxes": list(chunk.get("bboxes", []) or []),
        "chunking_strategy": str(chunk.get("chunking_strategy", "custom") or "custom"),
        "chunking_params": dict(chunk.get("chunking_params", {}) or {}),
    }
    return {k: v for k, v in row.items() if v not in (None, "", [], {})}


def write_user_chunks_jsonl(
    chunks: Iterable[Mapping],
    *,
    doc_id: str,
    output_path: str | Path,
    source_path: str | Path = "",
    source_sha1: Optional[str] = None,
) -> int:
    source_hash = source_sha1 or (sha1_file(source_path) if source_path else "")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for i, chunk in enumerate(chunks):
            row = make_user_chunk_row(
                chunk,
                doc_id=doc_id,
                source_path=source_path,
                source_sha1=source_hash,
                default_chunk_id=f"{doc_id}_user_c{i:06d}",
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _optional_int(value) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)
