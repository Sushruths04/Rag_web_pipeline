"""Resolve `chunk_id -> chunk_text` for the GT corpus.

The GT JSONL stores `supporting_spans[].chunk_id` but not the chunk text. RAGAS
needs the text strings as the `contexts` column. This module builds and reads a
JSONL cache that re-runs the canonical pipeline chunker over `data/docs/`,
keyed on `chunk_id`. The cache is regenerable; existing GT files are never
touched.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from rag_gt.core.types import QuestionGT, Span
from rag_gt.source_mapping import span_to_chunk_overlaps


_CHUNK_SUFFIX_RE = re.compile(r"_c(\d+)$")


def _normalized_key(chunk_id: str) -> str:
    """Map any `<doc>_c<digits>` chunk_id to a width-independent canonical form.

    The pipeline chunker has used both `_c{idx:04d}` and `_c{idx:06d}` over its
    history. GT JSONLs persisted before the format change (4-digit) must still
    resolve against caches built with the current chunker (6-digit). We canon-
    icalise on the integer index so width drift never breaks lookup.
    """
    m = _CHUNK_SUFFIX_RE.search(chunk_id)
    if not m:
        return chunk_id
    head = chunk_id[: m.start()]
    return f"{head}_c{int(m.group(1)):d}"


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    text: str
    char_start: int
    char_end: int
    sha1: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    source_path: str = ""
    source_sha1: str = ""
    extractor: str = ""

    def to_chunk_dict(self) -> dict:
        out = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "source_char_start": self.char_start,
            "source_char_end": self.char_end,
        }
        if self.page_start is not None:
            out["page_start"] = self.page_start
        if self.page_end is not None:
            out["page_end"] = self.page_end
        if self.page_start is not None and self.page_end is not None:
            out["pages"] = list(range(self.page_start, self.page_end + 1))
        return out


@dataclass
class CoverageReport:
    requested: int
    found: int
    missing: List[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return self.found / self.requested if self.requested else 1.0

    @property
    def is_complete(self) -> bool:
        return not self.missing


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _optional_int(value) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)


class ChunkResolver:
    def __init__(self, records: Dict[str, ChunkRecord]) -> None:
        self._records = records
        # Width-independent index so 4-digit GT IDs match 6-digit cache IDs
        # (and vice-versa).
        self._by_norm: Dict[str, ChunkRecord] = {}
        self._by_doc: Dict[str, List[ChunkRecord]] = {}
        for rec in records.values():
            self._by_norm[_normalized_key(rec.chunk_id)] = rec
            self._by_doc.setdefault(rec.doc_id, []).append(rec)
        for doc_records in self._by_doc.values():
            doc_records.sort(key=lambda r: (r.char_start, r.char_end, r.chunk_id))

    @classmethod
    def from_cache(
        cls, cache_path: Path | str = Path("data/cache/chunks.jsonl")
    ) -> "ChunkResolver":
        path = Path(cache_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Chunks cache not found: {path}. "
                f"Run `python -m rag_gt.cli.cache_chunks --input_dir data/docs "
                f"--output {path}` first."
            )
        records: Dict[str, ChunkRecord] = {}
        with open(path, "r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    rec = ChunkRecord(
                        chunk_id=d["chunk_id"],
                        doc_id=d["doc_id"],
                        text=d["text"],
                        char_start=int(d.get("char_start", 0)),
                        char_end=int(d.get("char_end", 0)),
                        sha1=d.get("sha1") or _sha1(d["text"]),
                        page_start=_optional_int(d.get("page_start")),
                        page_end=_optional_int(d.get("page_end")),
                        source_path=str(d.get("source_path", "") or ""),
                        source_sha1=str(d.get("source_sha1", "") or ""),
                        extractor=str(d.get("extractor", "") or ""),
                    )
                    records[rec.chunk_id] = rec
                except (json.JSONDecodeError, KeyError) as e:
                    raise ValueError(
                        f"Failed to parse chunks cache at {path}:{lineno}: {e}"
                    ) from e
        return cls(records)

    def _lookup(self, chunk_id: str) -> Optional[ChunkRecord]:
        rec = self._records.get(chunk_id)
        if rec is not None:
            return rec
        return self._by_norm.get(_normalized_key(chunk_id))

    def get(self, chunk_id: str) -> str:
        rec = self._lookup(chunk_id)
        if rec is None:
            raise KeyError(f"chunk_id not in resolver cache: {chunk_id!r}")
        return rec.text

    def get_many(self, chunk_ids: Iterable[str]) -> List[str]:
        out: List[str] = []
        for cid in chunk_ids:
            rec = self._lookup(cid)
            if rec is not None:
                out.append(rec.text)
        return out

    def record(self, chunk_id: str) -> Optional[ChunkRecord]:
        return self._lookup(chunk_id)

    def records_for_doc(self, doc_id: str) -> List[ChunkRecord]:
        return list(self._by_doc.get(doc_id, []))

    def chunks_for_span(
        self,
        span: Span,
        *,
        min_overlap_ratio: float = 0.0,
    ) -> List[str]:
        records = [r.to_chunk_dict() for r in self.records_for_doc(span.doc_id)]
        return [
            ov.chunk_id
            for ov in span_to_chunk_overlaps(
                span, records, min_overlap_ratio=min_overlap_ratio
            )
        ]

    def __contains__(self, chunk_id: str) -> bool:
        return self._lookup(chunk_id) is not None

    def __len__(self) -> int:
        return len(self._records)

    def required_for_gt(self, gt_questions: Iterable[QuestionGT]) -> Set[str]:
        out: Set[str] = set()
        for q in gt_questions:
            for f in q.required_facts:
                for s in f.supporting_spans:
                    if s.chunk_id:
                        out.add(s.chunk_id)
        return out

    def verify_coverage(
        self, gt_questions: Iterable[QuestionGT]
    ) -> CoverageReport:
        required = self.required_for_gt(gt_questions)
        missing = sorted(cid for cid in required if self._lookup(cid) is None)
        return CoverageReport(
            requested=len(required),
            found=len(required) - len(missing),
            missing=missing,
        )

    def verify_source_mapping(
        self, gt_questions: Iterable[QuestionGT]
    ) -> CoverageReport:
        """Verify source-anchored facts map to at least one active chunk.

        This is the v10 contract check: when GT spans contain canonical
        ``char_start`` / ``char_end`` offsets, the active chunk cache must expose
        compatible ranges so facts can be remapped independently of the original
        chunking strategy.
        """
        requested = 0
        missing: List[str] = []
        for q in gt_questions:
            for f in q.required_facts:
                for s in f.supporting_spans:
                    if s.char_start is None or s.char_end is None:
                        continue
                    requested += 1
                    if not self.chunks_for_span(s):
                        missing.append(f"{q.q_id}:{f.fact_id}:{s.doc_id}:{s.char_start}-{s.char_end}")
        return CoverageReport(
            requested=requested,
            found=requested - len(missing),
            missing=missing,
        )


def _list_input_paths(input_dir: Path) -> List[Path]:
    """Mirror gt_pipeline's input enumeration: PDF + DOCX, sorted, files only."""
    out: List[Path] = []
    for p in sorted(input_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() in (".pdf", ".docx"):
            out.append(p)
    return out


def build_chunks_cache(
    input_dir: Path | str,
    out_path: Path | str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    doc_type_hint: Optional[str] = None,
) -> int:
    """Re-run the canonical pipeline chunker and persist `chunk_id -> text`.

    Mirrors the chunking step from `pipeline.gt_pipeline` (ingest_document ->
    profile_document -> chunk_document) so chunk IDs are byte-identical to the
    ones embedded in GT supporting spans, provided the same chunk_size and
    chunk_overlap were used at GT-generation time.
    """
    # Local imports keep package import cheap (spaCy / pdfplumber are heavy).
    from rag_gt.chunking.strategies import chunk_document
    from rag_gt.ingestion import ingest_document
    from rag_gt.profiling.profiler import profile_document

    in_dir = Path(input_dir)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    paths = _list_input_paths(in_dir)
    written = 0
    tmp = out_file.with_suffix(out_file.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for p in paths:
            doc_type = doc_type_hint or "UNKNOWN"
            cached = None
            try:
                from rag_gt.pipeline.gt_pipeline import _cache_key, _load_doc_cache

                cached = _load_doc_cache(
                    _cache_key(str(p), doc_type, chunk_size, chunk_overlap)
                )
            except Exception:
                cached = None

            if cached is not None:
                doc, profile, _facts = cached
            else:
                try:
                    doc = ingest_document(str(p), doc_type=doc_type)
                except Exception as e:
                    # Skip docs that fail to ingest — same posture as retrieve_test.
                    print(f"[cache_chunks] skipped {p.name}: {e}")
                    continue
                profile = profile_document(doc, path=p)
            chunks = chunk_document(doc, profile, chunk_size, chunk_overlap)
            for c in chunks:
                rec = {
                    "chunk_id": c["chunk_id"],
                    "doc_id": c["doc_id"],
                    "text": c["text"],
                    "char_start": c.get("char_start", 0),
                    "char_end": c.get("char_end", 0),
                    "page_start": c.get("page_start"),
                    "page_end": c.get("page_end"),
                    "pages": c.get("pages", []),
                    "source_path": c.get("source_path", ""),
                    "source_sha1": c.get("source_sha1", ""),
                    "extractor": c.get("extractor", ""),
                    "source_units": c.get("source_units", []),
                    "sha1": _sha1(c["text"]),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
    tmp.replace(out_file)
    return written
