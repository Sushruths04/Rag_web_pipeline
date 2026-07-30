"""Map source-anchored GT facts onto an active user chunking strategy.

This is the V11 bridge between the RAG-GT ground-truth pipeline and an
external user's RAG pipeline. The user supplies chunks with source metadata;
RAG-GT maps each required fact span to those chunks without requiring the user
to adopt RAG-GT's own chunking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from rag_gt.core.types import Fact, QuestionGT, Span
from rag_gt.source_mapping import ChunkOverlap, span_to_chunk_overlaps
from rag_gt.storage.gt_io import load_gt


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _chunk_start(chunk: dict) -> Optional[int]:
    value = chunk.get("source_char_start", chunk.get("char_start"))
    if value in (None, ""):
        return None
    return int(value)


def _chunk_end(chunk: dict) -> Optional[int]:
    value = chunk.get("source_char_end", chunk.get("char_end"))
    if value in (None, ""):
        return None
    return int(value)


def _normalize_chunk_row(chunk: dict) -> dict:
    row = dict(chunk)
    if "chunk_id" not in row and "id" in row:
        row["chunk_id"] = row["id"]
    start = _chunk_start(row)
    end = _chunk_end(row)
    if start is not None:
        row["char_start"] = start
        row["source_char_start"] = start
    if end is not None:
        row["char_end"] = end
        row["source_char_end"] = end
    pages = row.get("pages")
    if not pages and row.get("page_start") not in (None, "") and row.get("page_end") not in (None, ""):
        row["pages"] = list(range(int(row["page_start"]), int(row["page_end"]) + 1))
    return row


def _chunks_by_doc(chunks: Iterable[dict]) -> Dict[str, List[dict]]:
    by_doc: Dict[str, List[dict]] = {}
    for raw in chunks:
        c = _normalize_chunk_row(raw)
        doc_id = str(c.get("doc_id", "") or "")
        if not doc_id or not c.get("chunk_id"):
            continue
        by_doc.setdefault(doc_id, []).append(c)
    for rows in by_doc.values():
        rows.sort(
            key=lambda c: (
                int(c.get("source_char_start", c.get("char_start", 0)) or 0),
                int(c.get("source_char_end", c.get("char_end", 0)) or 0),
                str(c.get("chunk_id", "")),
            )
        )
    return by_doc


def _span_source_id(span: Span) -> str:
    if span.source_sha1:
        return f"sha1:{span.source_sha1}"
    if span.source_path:
        return f"path:{Path(span.source_path).name}"
    return f"doc:{span.doc_id}"


def _mapping_method(span: Span, overlap: Optional[ChunkOverlap]) -> str:
    if overlap is not None and span.source_sha1:
        return "source_sha1_char_overlap"
    if overlap is not None:
        return "doc_id_char_overlap"
    return "unmapped"


def _page_fuzzy_overlaps(
    fact_text: str,
    span: Span,
    chunks: List[dict],
    *,
    threshold: float,
) -> List[ChunkOverlap]:
    if span.page_start is None and span.page_end is None:
        return []
    try:
        from rapidfuzz import fuzz
    except Exception:
        return []
    page_start = span.page_start or span.page_end
    page_end = span.page_end or span.page_start
    if page_start is None or page_end is None:
        return []
    out: List[ChunkOverlap] = []
    for chunk in chunks:
        c_page_start = chunk.get("page_start")
        c_page_end = chunk.get("page_end", c_page_start)
        if c_page_start in (None, "") or c_page_end in (None, ""):
            continue
        if int(c_page_end) < page_start or int(c_page_start) > page_end:
            continue
        ratio = float(fuzz.partial_ratio(fact_text, str(chunk.get("text", "") or "")))
        if ratio < threshold:
            continue
        cs = _chunk_start(chunk) or 0
        ce = _chunk_end(chunk) or cs
        out.append(
            ChunkOverlap(
                chunk_id=str(chunk.get("chunk_id", "")),
                doc_id=span.doc_id,
                char_start=cs,
                char_end=ce,
                overlap_chars=max(1, ce - cs),
                overlap_ratio=ratio / 100.0,
                pages=list(range(int(c_page_start), int(c_page_end) + 1)),
            )
        )
    out.sort(key=lambda x: (-x.overlap_ratio, x.char_start, x.chunk_id))
    return out


def _map_fact(
    q: QuestionGT,
    fact: Fact,
    chunks_for_doc: Dict[str, List[dict]],
    *,
    min_overlap_ratio: float,
    fallback_fuzzy_threshold: float,
) -> dict:
    all_overlaps: List[ChunkOverlap] = []
    span_rows: List[dict] = []
    for span in fact.supporting_spans:
        doc_chunks = chunks_for_doc.get(span.doc_id, [])
        overlaps = span_to_chunk_overlaps(
            span, doc_chunks, min_overlap_ratio=min_overlap_ratio
        )
        method = _mapping_method(span, overlaps[0] if overlaps else None)
        if not overlaps:
            overlaps = _page_fuzzy_overlaps(
                fact.text,
                span,
                doc_chunks,
                threshold=fallback_fuzzy_threshold,
            )
            if overlaps:
                method = "page_fuzzy_text"
        all_overlaps.extend(overlaps)
        span_rows.append(
            {
                "doc_id": span.doc_id,
                "source_id": _span_source_id(span),
                "char_start": span.char_start,
                "char_end": span.char_end,
                "page_start": span.page_start,
                "page_end": span.page_end,
                "matched_chunk_ids": [o.chunk_id for o in overlaps],
                "best_chunk_id": overlaps[0].chunk_id if overlaps else "",
                "best_overlap_ratio": overlaps[0].overlap_ratio if overlaps else 0.0,
                "mapping_method": method,
            }
        )

    best_by_chunk: Dict[str, ChunkOverlap] = {}
    for overlap in all_overlaps:
        current = best_by_chunk.get(overlap.chunk_id)
        if current is None or overlap.overlap_ratio > current.overlap_ratio:
            best_by_chunk[overlap.chunk_id] = overlap
    ordered = sorted(
        best_by_chunk.values(), key=lambda o: (-o.overlap_ratio, o.char_start, o.chunk_id)
    )

    best = ordered[0] if ordered else None
    return {
        "q_id": q.q_id,
        "fact_id": fact.fact_id,
        "fact_text": fact.text,
        "role": fact.role,
        "doc_ids": sorted({s.doc_id for s in fact.supporting_spans}),
        "matched_chunk_ids": [o.chunk_id for o in ordered],
        "best_chunk_id": best.chunk_id if best else "",
        "best_overlap_ratio": best.overlap_ratio if best else 0.0,
        "mapping_method": span_rows[0]["mapping_method"] if span_rows else "unmapped",
        "spans": span_rows,
        "source_page_start": _min_optional(s.page_start for s in fact.supporting_spans),
        "source_page_end": _max_optional(s.page_end for s in fact.supporting_spans),
        "char_start": _min_optional(s.char_start for s in fact.supporting_spans),
        "char_end": _max_optional(s.char_end for s in fact.supporting_spans),
    }


def _min_optional(values: Iterable[Optional[int]]) -> Optional[int]:
    cleaned = [int(v) for v in values if v is not None]
    return min(cleaned) if cleaned else None


def _max_optional(values: Iterable[Optional[int]]) -> Optional[int]:
    cleaned = [int(v) for v in values if v is not None]
    return max(cleaned) if cleaned else None


def _question_map(q: QuestionGT, fact_rows: List[dict], chunk_profile_id: str) -> dict:
    required_chunk_ids: List[str] = []
    seen: set[str] = set()
    mapped_fact_ids: List[str] = []
    unmapped_fact_ids: List[str] = []
    for row in fact_rows:
        if row["matched_chunk_ids"]:
            mapped_fact_ids.append(row["fact_id"])
        else:
            unmapped_fact_ids.append(row["fact_id"])
        for cid in row["matched_chunk_ids"]:
            if cid not in seen:
                seen.add(cid)
                required_chunk_ids.append(cid)
    return {
        "q_id": q.q_id,
        "chunk_profile_id": chunk_profile_id,
        "required_fact_ids": list(q.required_fact_ids),
        "mapped_fact_ids": mapped_fact_ids,
        "unmapped_fact_ids": unmapped_fact_ids,
        "required_fact_groups": q.required_fact_groups,
        "required_chunk_ids": required_chunk_ids,
        "required_chunk_count": len(required_chunk_ids),
        "joint_required": len(q.required_fact_ids) > 1,
        "hop_type": q.hop_type,
        "reasoning_depth": q.difficulty_reasoning_depth,
        "mapping_complete": not unmapped_fact_ids,
    }


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _summary(
    *,
    gt_path: Path,
    chunks_path: Path,
    chunk_profile_id: str,
    questions: List[QuestionGT],
    chunks: List[dict],
    fact_rows: List[dict],
    question_rows: List[dict],
) -> dict:
    mapped_facts = sum(1 for r in fact_rows if r["matched_chunk_ids"])
    strong_mapped = sum(
        1 for r in fact_rows
        if r["mapping_method"] in {"source_sha1_char_overlap", "doc_id_char_overlap"}
        and r["matched_chunk_ids"]
    )
    complete_questions = sum(1 for r in question_rows if r["mapping_complete"])
    distinct_chunks = sorted({cid for r in fact_rows for cid in r["matched_chunk_ids"]})
    return {
        "gt": str(gt_path),
        "chunks": str(chunks_path),
        "chunk_profile_id": chunk_profile_id,
        "n_questions": len(questions),
        "n_chunks": len(chunks),
        "n_facts": len(fact_rows),
        "mapped_facts": mapped_facts,
        "unmapped_facts": len(fact_rows) - mapped_facts,
        "fact_mapping_coverage": mapped_facts / len(fact_rows) if fact_rows else 1.0,
        "strong_mapped_facts": strong_mapped,
        "strong_mapping_coverage": strong_mapped / len(fact_rows) if fact_rows else 1.0,
        "questions_complete": complete_questions,
        "question_mapping_coverage": complete_questions / len(question_rows)
        if question_rows else 1.0,
        "distinct_required_chunks": len(distinct_chunks),
        "mapping_methods": _counts(r["mapping_method"] for r in fact_rows),
    }


def _counts(values: Iterable[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def main() -> None:
    p = argparse.ArgumentParser(
        prog="rag-gt-map-facts-to-chunks",
        description="Map RAG-GT source-anchored facts to an external chunk profile.",
    )
    p.add_argument("--gt", required=True, help="Source-anchored GT JSONL.")
    p.add_argument("--chunks", required=True, help="User/RAG chunks JSONL.")
    p.add_argument("--chunk-profile-id", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=0.0,
        help="Minimum fraction of a fact span that must overlap a chunk.",
    )
    p.add_argument(
        "--fallback-fuzzy-threshold",
        type=float,
        default=85.0,
        help="Weak fallback: page-overlap + fuzzy fact text match threshold.",
    )
    args = p.parse_args()

    gt_path = Path(args.gt)
    chunks_path = Path(args.chunks)
    out_dir = Path(args.output_dir)

    questions = load_gt(gt_path.stem, in_dir=gt_path.parent)
    chunks = _load_jsonl(chunks_path)
    chunks_for_doc = _chunks_by_doc(chunks)

    fact_rows: List[dict] = []
    question_rows: List[dict] = []
    for q in questions:
        q_fact_rows = [
            _map_fact(
                q, fact, chunks_for_doc,
                min_overlap_ratio=args.min_overlap_ratio,
                fallback_fuzzy_threshold=args.fallback_fuzzy_threshold,
            )
            for fact in q.required_facts
        ]
        fact_rows.extend(q_fact_rows)
        question_rows.append(_question_map(q, q_fact_rows, args.chunk_profile_id))

    fact_path = out_dir / "fact_chunk_map.jsonl"
    question_path = out_dir / "question_chunk_map.jsonl"
    summary_path = out_dir / "mapping_summary.json"
    _write_jsonl(fact_path, fact_rows)
    _write_jsonl(question_path, question_rows)
    summary = _summary(
        gt_path=gt_path,
        chunks_path=chunks_path,
        chunk_profile_id=args.chunk_profile_id,
        questions=questions,
        chunks=chunks,
        fact_rows=fact_rows,
        question_rows=question_rows,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[map_facts_to_chunks] wrote {fact_path}")
    print(f"[map_facts_to_chunks] wrote {question_path}")
    print(f"[map_facts_to_chunks] wrote {summary_path}")
    print(
        "[map_facts_to_chunks] coverage "
        f"facts={summary['fact_mapping_coverage']:.1%}, "
        f"questions={summary['question_mapping_coverage']:.1%}"
    )


if __name__ == "__main__":
    main()
