"""Fact store: resolve `fact_id -> Fact` independently of GT format.

Two GT formats live in this repo:
  - new format: `required_facts` embedded in each GT row with text + supporting_spans
  - old format: only `required_fact_ids` (just strings)

The fact store unifies both. For new-format GT it harvests facts inline; for
old-format GT it re-runs the pipeline's fact extraction on the source documents
and persists a JSONL cache keyed by `fact_id`. After that the lookup is a flat
dict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set

from rag_gt.core.types import Fact, Span


@dataclass
class StoredFact:
    """Wire-format mirror of `core.types.Fact` (without runtime validators)."""

    fact_id: str
    text: str
    role: str
    doc_id: str
    supporting_spans: List[Dict[str, object]] = field(default_factory=list)

    @classmethod
    def from_fact(cls, f: Fact, doc_id: Optional[str] = None) -> "StoredFact":
        spans = [s.to_dict() for s in f.supporting_spans]
        return cls(
            fact_id=f.fact_id,
            text=f.text,
            role=f.role,
            doc_id=doc_id or (spans[0]["doc_id"] if spans else ""),
            supporting_spans=spans,
        )

    def to_fact(self) -> Fact:
        return Fact(
            fact_id=self.fact_id,
            text=self.text,
            role=self.role,
            supporting_spans=[
                Span.from_any(s)
                for s in self.supporting_spans
            ],
        )


class FactStore:
    """Append-only resolver. Build once, query many times."""

    def __init__(self, facts: Optional[Dict[str, StoredFact]] = None) -> None:
        self._facts: Dict[str, StoredFact] = dict(facts or {})

    # ---------- factories ----------

    @classmethod
    def from_gt_inline(cls, gt_path: Path | str) -> "FactStore":
        """Harvest facts embedded inline in a new-format GT JSONL."""
        store = cls()
        path = Path(gt_path)
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                row = json.loads(line)
                doc_id = (row.get("doc_ids") or [""])[0]
                for f_dict in row.get("required_facts") or []:
                    sf = StoredFact(
                        fact_id=f_dict["fact_id"],
                        text=f_dict["text"],
                        role=f_dict["role"],
                        doc_id=doc_id,
                        supporting_spans=list(f_dict.get("supporting_spans") or []),
                    )
                    store._facts[sf.fact_id] = sf
        return store

    @classmethod
    def from_cache(cls, cache_path: Path | str) -> "FactStore":
        path = Path(cache_path)
        if not path.exists():
            raise FileNotFoundError(f"Fact store cache not found: {path}")
        store = cls()
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                d = json.loads(line)
                sf = StoredFact(
                    fact_id=d["fact_id"],
                    text=d["text"],
                    role=d["role"],
                    doc_id=d.get("doc_id", ""),
                    supporting_spans=list(d.get("supporting_spans") or []),
                )
                store._facts[sf.fact_id] = sf
        return store

    @classmethod
    def build_from_source(
        cls,
        input_dir: Path | str,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> "FactStore":
        """Re-run the pipeline's facts step over source PDFs/DOCX.

        Mirrors `pipeline.gt_pipeline`: ingest -> profile -> chunk ->
        extract_candidate_facts -> find_fact_spans -> drop facts without spans.
        Produces the full fact corpus the GT was originally drawn from.
        """
        # Local imports keep the package-import path light.
        from rag_gt.chunking.strategies import chunk_document
        from rag_gt.facts.extraction import extract_candidate_facts
        from rag_gt.ingestion import ingest_document
        from rag_gt.profiling.profiler import profile_document
        from rag_gt.spans.normalization import find_fact_spans, tokenize_document

        in_dir = Path(input_dir)
        store = cls()
        for path in sorted(in_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in (".pdf", ".docx"):
                continue
            cached = None
            try:
                from rag_gt.pipeline.gt_pipeline import _cache_key, _load_doc_cache

                cached = _load_doc_cache(
                    _cache_key(str(path), "UNKNOWN", chunk_size, chunk_overlap)
                )
            except Exception:
                cached = None

            if cached is not None:
                doc, _profile, facts = cached
            else:
                try:
                    doc = ingest_document(str(path))
                except Exception as e:
                    print(f"[fact_store] skipped {path.name}: {e}")
                    continue
                profile = profile_document(doc, path=path)
                chunks = chunk_document(doc, profile, chunk_size, chunk_overlap)
                facts = extract_candidate_facts(doc, chunks, llm=None)
                doc_toks = tokenize_document(doc)
                facts = find_fact_spans(doc, doc_toks, facts, chunks)
                facts = [f for f in facts if f.supporting_spans]
            for f in facts:
                store._facts[f.fact_id] = StoredFact.from_fact(f, doc_id=doc.doc_id)
            print(f"[fact_store] {doc.doc_id}: {len(facts)} facts")
        return store

    # ---------- merge ----------

    def merge(self, other: "FactStore") -> "FactStore":
        merged = dict(self._facts)
        merged.update(other._facts)
        return FactStore(merged)

    # ---------- queries ----------

    def get(self, fact_id: str) -> Fact:
        sf = self._facts.get(fact_id)
        if sf is None:
            raise KeyError(f"fact_id not in store: {fact_id!r}")
        return sf.to_fact()

    def get_optional(self, fact_id: str) -> Optional[Fact]:
        sf = self._facts.get(fact_id)
        return sf.to_fact() if sf else None

    def __contains__(self, fact_id: str) -> bool:
        return fact_id in self._facts

    def __len__(self) -> int:
        return len(self._facts)

    def __iter__(self) -> Iterator[str]:
        return iter(self._facts)

    def coverage(self, required_fact_ids: Iterable[str]) -> "FactStoreCoverage":
        req = list(required_fact_ids)
        missing = sorted(fid for fid in req if fid not in self._facts)
        return FactStoreCoverage(
            requested=len(req),
            found=len(req) - len(missing),
            missing=missing,
        )

    # ---------- persistence ----------

    def save(self, cache_path: Path | str) -> Path:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for fid in sorted(self._facts):
                sf = self._facts[fid]
                f.write(
                    json.dumps(
                        {
                            "fact_id": sf.fact_id,
                            "text": sf.text,
                            "role": sf.role,
                            "doc_id": sf.doc_id,
                            "supporting_spans": sf.supporting_spans,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        tmp.replace(path)
        return path


@dataclass
class FactStoreCoverage:
    requested: int
    found: int
    missing: List[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return self.found / self.requested if self.requested else 1.0

    @property
    def is_complete(self) -> bool:
        return not self.missing


def required_fact_ids_from_gt(gt_path: Path | str) -> Set[str]:
    """Return the union of `required_fact_ids` across all rows in a GT JSONL."""
    out: Set[str] = set()
    with open(gt_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            d = json.loads(line)
            for fid in d.get("required_fact_ids") or []:
                out.add(fid)
    return out
