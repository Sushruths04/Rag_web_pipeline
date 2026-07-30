"""Stage 3 — SFU-grade fact extraction.

Converts Stage-2 chunks into Fact objects with:
- canonical_form: LLM rewrite that resolves anaphora, NLI-entailed by source
- self_containment_score: 0..1 LLM-scored; self_containment_known=True when set
- supporting_spans: bbox + page provenance inherited from the parent chunk

Two extraction paths:
1. LLM (primary): ``segment_chunk_into_sfus`` segments chunks into
   semantically-coherent spans, then canonical_form_rewrite + score each span.
2. Rule-based (fallback): spaCy sentence splitting when LLM is None or fails.

Both produce the same Fact shape so downstream stages are path-agnostic.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from loguru import logger

from rag_gt.allpdf.chunk import ChunkResult
from rag_gt.allpdf.preflight import DocProfile
from rag_gt.core.types import Fact, Span, SourceBBox
from rag_gt.facts.roles import detect_role


def _fact_to_cache(f: Fact) -> dict:
    """Serialize a Fact for the per-chunk extraction cache (resume support)."""
    return {
        "fact_id": f.fact_id, "text": f.text, "raw_text": f.raw_text,
        "canonical_form": f.canonical_form,
        "self_containment_score": f.self_containment_score,
        "self_containment_known": getattr(f, "self_containment_known", False),
        "role": str(f.role), "weight": f.weight,
        "spans": [s.to_dict() for s in (f.supporting_spans or [])],
    }


def _fact_from_cache(d: dict) -> Fact:
    return Fact(
        fact_id=d["fact_id"], text=d["text"], raw_text=d.get("raw_text") or d["text"],
        canonical_form=d.get("canonical_form") or d["text"],
        self_containment_score=d.get("self_containment_score", 1.0),
        self_containment_known=d.get("self_containment_known", False),
        role=d.get("role", "descriptive"), weight=d.get("weight", 0.5),
        supporting_spans=[Span.from_any(s) for s in d.get("spans", [])],
    )

if TYPE_CHECKING:
    from rag_gt.core.llm import LLM


MIN_SPAN_CHARS = 30
MIN_SELF_CONTAINMENT = 0.5

# BUG-1: ISO PDF extraction flattens clause lists, leaving fact spans that begin
# or end with an orphan list/clause marker ("— Wire speed feed range ...",
# "For multiple electrode systems, ... polarity. —", "4.4.6 Back gouging").
# Strip those markers so the canonical-form rewrite receives clean input and the
# fact never ships as a bare bullet. (The rewrite then resolves the remaining
# noun phrase into a full sentence using the surrounding clause context.)
_LEADING_LIST_MARKER_RE = re.compile(r"^\s*(?:[—–•*]|[-]{1,2}|[a-z]\))\s+")
_TRAILING_LIST_MARKER_RE = re.compile(r"\s*[—–•*]\s*$")
_LEADING_CLAUSE_NUM_RE = re.compile(r"^\s*\d+(?:\.\d+){1,}\s+")

# Short-fact expansion: facts under this word count lack enough context to be
# meaningful on their own ("A WPS may cover a group of materials." = 8 words).
# We find the most matching neighbouring sentence in the source chunk and
# prepend/append it so the fact carries its full meaning.
_SHORT_FACT_THRESHOLD = 12
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _clean_list_markers(text: str) -> str:
    """Remove orphan leading/trailing bullet markers and clause-number prefixes."""
    t = (text or "").strip()
    t = _LEADING_CLAUSE_NUM_RE.sub("", t)
    t = _LEADING_LIST_MARKER_RE.sub("", t)
    t = _TRAILING_LIST_MARKER_RE.sub("", t)
    return t.strip()


def _expand_short_fact(canonical: str, chunk_text: str) -> str:
    """If canonical is under _SHORT_FACT_THRESHOLD words, prepend the best
    neighbouring sentence from chunk_text to supply the missing context.

    ISO standard text commonly splits a lead-in sentence ("The following shall be
    specified:") from its sub-facts ("For manual welding, the maximum run width.").
    Without the lead-in the sub-fact has no grammatical subject and reads as a
    fragment. We find the sentence in the chunk that most overlaps with the canonical
    form (that is the fact itself), then attach the sentence before it (or after, if
    there is no predecessor) to restore its meaning.
    """
    words = canonical.split()
    if len(words) >= _SHORT_FACT_THRESHOLD:
        return canonical

    # Split chunk into sentences; keep only ones that are substantial.
    raw_sents = [s.strip() for s in _SENT_SPLIT_RE.split(chunk_text) if s.strip()]
    usable = [
        s for s in raw_sents
        if len(s.split()) >= 5 and not _LEADING_LIST_MARKER_RE.match(s)
    ]
    if not usable:
        return canonical

    # Find which usable sentence is the canonical (highest word overlap).
    canon_words = set(w.lower() for w in words if len(w) > 2)
    best_idx, best_overlap = -1, 0
    for i, sent in enumerate(usable):
        sent_words = set(w.lower() for w in sent.split() if len(w) > 2)
        overlap = len(canon_words & sent_words)
        if overlap > best_overlap:
            best_overlap, best_idx = overlap, i

    if best_idx == -1 or best_overlap == 0:
        return canonical

    # Prefer the predecessor (lead-in context is before the sub-fact in ISO text).
    if best_idx > 0:
        pred = usable[best_idx - 1]
        if pred.lower().rstrip(".!?") != canonical.lower().rstrip(".!?"):
            joiner = " " if pred.endswith((":", ";")) else ". "
            return pred.rstrip(" ") + joiner + canonical

    # Fallback: attach the successor sentence.
    if best_idx < len(usable) - 1:
        succ = usable[best_idx + 1]
        if succ.lower().rstrip(".!?") != canonical.lower().rstrip(".!?"):
            return canonical.rstrip(" ") + " " + succ

    return canonical


@dataclass
class ExtractResult:
    doc_id: str
    n_facts: int
    n_canonical: int        # facts where canonical_form != raw_text
    n_weak_containment: int # scored facts below MIN_SELF_CONTAINMENT
    extraction_path: str    # "llm" | "rule_based" | "mixed"
    wall_sec: float
    facts: List[Fact] = field(default_factory=list)
    # Per-chunk timing (observability)
    n_llm_chunks: int = 0
    n_rule_chunks: int = 0
    llm_total_sec: float = 0.0
    rule_total_sec: float = 0.0
    llm_chunk_sec_list: List[float] = field(default_factory=list)  # time per LLM chunk

    def summary(self) -> dict:
        avg_llm = (
            round(self.llm_total_sec / self.n_llm_chunks, 2)
            if self.n_llm_chunks else 0.0
        )
        return {
            "doc_id": self.doc_id,
            "n_facts": self.n_facts,
            "n_canonical": self.n_canonical,
            "n_weak_containment": self.n_weak_containment,
            "extraction_path": self.extraction_path,
            "wall_sec": round(self.wall_sec, 2),
            "n_llm_chunks": self.n_llm_chunks,
            "n_rule_chunks": self.n_rule_chunks,
            "llm_total_sec": round(self.llm_total_sec, 2),
            "rule_total_sec": round(self.rule_total_sec, 2),
            "llm_avg_sec_per_chunk": avg_llm,
        }


def _make_span(chunk: dict, doc_id: str, char_start: int, char_end: int) -> Span:
    """Build a Span carrying page + bbox provenance from a Stage-2 chunk dict."""
    bboxes = [SourceBBox.from_any(b) for b in chunk.get("bboxes", [])]
    return Span(
        doc_id=doc_id,
        chunk_id=chunk.get("chunk_id", ""),
        start_token=0,
        end_token=0,
        char_start=char_start,
        char_end=char_end,
        page_start=chunk.get("page_start"),
        page_end=chunk.get("page_end"),
        bboxes=bboxes,
    )


def _rule_based_facts(doc_id: str, chunks: List[dict], id_offset: int = 0) -> List[Fact]:
    """Sentence-split each chunk without LLM. Fast fallback path."""
    from rag_gt.chunking.strategies import _get_nlp

    from rag_gt.facts.domain_filter import strip_running_artifacts

    nlp = _get_nlp()
    facts: List[Fact] = []
    ctr = id_offset

    for chunk in chunks:
        text = strip_running_artifacts(chunk.get("text", "").strip())
        if not text:
            continue
        char_base = chunk.get("char_start") or 0

        try:
            spacy_doc = nlp(text)
            sents = [(s.text.strip(), s.start_char, s.end_char) for s in spacy_doc.sents
                     if len(s.text.strip()) >= MIN_SPAN_CHARS]
        except Exception:
            sents = [(text, 0, len(text))] if len(text) >= MIN_SPAN_CHARS else []

        for sent_text, sc, ec in sents:
            ctr += 1
            # Expand short sentences using the surrounding chunk text for context.
            expanded = _expand_short_fact(sent_text, text)
            span = _make_span(chunk, doc_id, char_base + sc, char_base + ec)
            try:
                facts.append(Fact(
                    fact_id=f"{doc_id}_F{ctr:06d}",
                    text=expanded,
                    raw_text=sent_text,
                    canonical_form=expanded,
                    self_containment_score=1.0,
                    self_containment_known=False,
                    role=detect_role(sent_text),
                    weight=0.5,
                    supporting_spans=[span],
                ))
            except ValueError as e:
                logger.debug(f"[extract] skipping fact (validation): {e}")
    return facts


def _llm_facts_for_chunk(
    doc_id: str,
    chunk: dict,
    chunk_idx: int,
    llm: "LLM",
    nli_model,
) -> List[Fact]:
    """Extract SFU facts from one chunk via LLM segmentation + canonical rewrite."""
    from rag_gt.facts.semantic_extraction import (
        USE_MODULE_NLI,
        segment_chunk_into_sfus,
        canonical_form_rewrite,
        self_containment_score as score_containment,
    )
    from rag_gt.facts.domain_filter import strip_running_artifacts

    # RC-4 root fix: remove controlled-copy / running-header watermarks before
    # the SFU segmenter sees the chunk, so a page header never folds into a fact.
    text = strip_running_artifacts(chunk.get("text", "").strip())
    if not text:
        return []

    char_base = chunk.get("char_start") or 0

    try:
        segments = segment_chunk_into_sfus(text, llm)
    except Exception as e:
        logger.warning(f"[extract] LLM segmenter failed chunk {chunk_idx}: {e}")
        return []

    if not segments:
        # Treat whole chunk as one span.
        spans_iter = [(text, 0, len(text))]
    else:
        spans_iter = [(s.text, s.start_char, s.end_char) for s in segments]

    facts: List[Fact] = []
    base_id = chunk_idx * 1000

    for span_idx, (span_text, sc, ec) in enumerate(spans_iter):
        if len(span_text.strip()) < MIN_SPAN_CHARS:
            continue

        # BUG-1: strip orphan list/clause markers so the rewrite gets clean input
        # and the fact never ships as a bare bullet ("— Wire speed feed range ...").
        span_text = _clean_list_markers(span_text)
        if len(span_text) < MIN_SPAN_CHARS:
            continue

        # Canonical-form rewrite with surrounding chunk as context. Pass the
        # USE_MODULE_NLI sentinel so the rewrite is NLI-verified against the
        # source (a rewrite that changes meaning is rejected and the original
        # wording is kept) instead of being accepted with a synthetic 1.0.
        # ``nli_model`` from the caller takes precedence when supplied.
        try:
            rewrite = canonical_form_rewrite(
                span_text, text, llm, nli_model or USE_MODULE_NLI
            )
            canon = rewrite.canonical_form
        except Exception as e:
            logger.warning(f"[extract] canonical rewrite failed span {span_idx}: {e}")
            canon = span_text

        # NONE-guard: if the rewrite judged the span to be a blank form/template
        # region (returns "NONE"), it carries no factual claim — drop it.
        if canon.strip().rstrip(".").upper() == "NONE" or len(canon.strip()) < MIN_SPAN_CHARS:
            logger.debug(f"[extract] dropping non-factual span: {span_text[:60]!r}")
            continue

        # If the canonical form is still very short (< _SHORT_FACT_THRESHOLD words),
        # attach the most relevant neighbouring sentence from the chunk for context.
        original_len = len(canon.split())
        canon = _expand_short_fact(canon, text)
        if len(canon.split()) != original_len:
            logger.debug(f"[extract] expanded short fact ({original_len}w→{len(canon.split())}w): {canon[:80]!r}")

        # Self-containment score.
        try:
            sc_score = score_containment(canon, llm)
            sc_known = True
        except Exception as e:
            logger.warning(f"[extract] self_containment_score failed: {e}")
            sc_score = 1.0
            sc_known = False

        # weight must be in (0, 2].
        weight = max(0.01, float(sc_score)) if sc_known else 0.5

        fact_id = f"{doc_id}_F{base_id + span_idx + 1:06d}"
        span = _make_span(chunk, doc_id, char_base + sc, char_base + ec)
        try:
            facts.append(Fact(
                fact_id=fact_id,
                text=canon,         # canonical form is the primary text
                raw_text=span_text,
                canonical_form=canon,
                self_containment_score=sc_score,
                self_containment_known=sc_known,
                role=detect_role(span_text),
                weight=weight,
                supporting_spans=[span],
            ))
        except ValueError as e:
            logger.debug(f"[extract] skipping fact (validation): {e}")

    return facts


def extract_sfu_facts(
    profile: DocProfile,
    chunk_result: ChunkResult,
    llm: Optional["LLM"] = None,
    nli_model=None,
    *,
    max_chunks: Optional[int] = None,
    llm_chunk_cap: Optional[int] = 40,
    cache_dir: Optional[str] = None,
    workers: Optional[int] = None,
) -> ExtractResult:
    """Extract SFU facts from Stage-2 chunks.

    ``llm_chunk_cap``: max chunks through LLM path per doc (cap for cost control).
    ``max_chunks``: hard cap on total chunks processed (smoke tests).
    ``cache_dir``: if set, each LLM chunk's facts are cached to
        ``cache_dir/chunk_NNNN.json`` and reused on re-run — makes long extraction
        resumable across interruptions (run repeatedly until all chunks cached).
    ``workers``: parallelism for the LLM path (BUG-A). Defaults to
        ``performance.max_concurrent_llm_calls`` from config. Fact IDs are derived
        from chunk position (``chunk_idx*1000+span_idx``), not arrival order, so the
        parallel result is identical to the sequential one; the rule-based fallback
        for empty/failed chunks is applied in a deterministic in-order second pass.
        Set ``workers=1`` to force the old sequential path.
    """
    _cache = Path(cache_dir) if cache_dir else None
    if _cache:
        _cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    doc_id = profile.doc_id
    chunks = chunk_result.chunks

    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    use_llm = llm is not None
    if use_llm and llm_chunk_cap is not None and len(chunks) > llm_chunk_cap:
        llm_cap = llm_chunk_cap
        logger.info(
            f"[extract] {doc_id}: capping LLM path at {llm_cap}/{len(chunks)} chunks"
        )
    else:
        llm_cap = len(chunks)

    facts: List[Fact] = []
    llm_failures = 0
    llm_chunk_times: List[float] = []

    if workers is None:
        try:
            from rag_gt.core.config import load_config
            workers = int(load_config()["performance"]["max_concurrent_llm_calls"])
        except Exception:
            workers = 1
    workers = max(1, int(workers))

    if use_llm:
        t_llm_start = time.time()
        llm_chunks = list(enumerate(chunks[:llm_cap]))

        def _process_chunk(i: int, chunk: dict) -> tuple[int, str, object]:
            """Return (chunk_idx, kind, payload). kind in cached|llm|error.

            Only the LLM call happens here (parallel-safe: fact IDs are
            position-derived and the shared NLI model is serialized behind a lock
            in nli_check). The rule-based fallback for empty/error chunks is NOT
            done here — it needs the running fact count for deterministic IDs and
            is applied in the in-order pass below.
            """
            cache_file = (_cache / f"chunk_{i:04d}.json") if _cache else None
            if cache_file and cache_file.exists():
                try:
                    cached = json.loads(cache_file.read_text(encoding="utf-8"))
                    return i, "cached", [_fact_from_cache(d) for d in cached]
                except Exception as e:
                    logger.warning(f"[extract] {doc_id} chunk {i} cache read failed: {e}")
            tc = time.time()
            try:
                chunk_facts = _llm_facts_for_chunk(doc_id, chunk, i, llm, nli_model)
                return i, "llm", (chunk_facts, round(time.time() - tc, 2))
            except Exception as e:
                logger.warning(f"[extract] {doc_id} chunk {i} LLM path failed: {e}")
                return i, "error", round(time.time() - tc, 2)

        # Phase 1 — parallel LLM calls (or sequential when workers == 1).
        per_chunk: dict[int, tuple[str, object]] = {}
        if workers == 1:
            for i, chunk in llm_chunks:
                _, kind, payload = _process_chunk(i, chunk)
                per_chunk[i] = (kind, payload)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for i, kind, payload in pool.map(
                    lambda ic: _process_chunk(ic[0], ic[1]), llm_chunks
                ):
                    per_chunk[i] = (kind, payload)

        # Phase 2 — deterministic in-order assembly + rule-based fallback + cache.
        for i, chunk in llm_chunks:
            kind, payload = per_chunk[i]
            if kind == "cached":
                facts.extend(payload)  # already cached; no re-write, no timing
                continue
            if kind == "error":
                llm_failures += 1
                chunk_facts = _rule_based_facts(doc_id, [chunk], id_offset=len(facts))
                llm_chunk_times.append(float(payload))
            else:  # "llm"
                llm_facts, elapsed = payload
                chunk_facts = llm_facts or _rule_based_facts(
                    doc_id, [chunk], id_offset=len(facts)
                )
                llm_chunk_times.append(float(elapsed))
            facts.extend(chunk_facts)
            cache_file = (_cache / f"chunk_{i:04d}.json") if _cache else None
            if cache_file:
                try:
                    cache_file.write_text(
                        json.dumps([_fact_to_cache(f) for f in chunk_facts], ensure_ascii=False),
                        encoding="utf-8",
                    )
                except Exception as e:
                    logger.warning(f"[extract] {doc_id} chunk {i} cache write failed: {e}")
        llm_total = time.time() - t_llm_start

        # Rule-based for remainder beyond llm_cap.
        t_rb_start = time.time()
        if llm_cap < len(chunks):
            rb = _rule_based_facts(doc_id, chunks[llm_cap:], id_offset=len(facts))
            facts.extend(rb)
        rule_total = time.time() - t_rb_start
        n_rule = max(0, len(chunks) - llm_cap)

        extraction_path = (
            "rule_based" if llm_cap == 0 else
            "mixed" if (llm_failures > 0 or llm_cap < len(chunks)) else "llm"
        )
    else:
        t_rb_start = time.time()
        facts = _rule_based_facts(doc_id, chunks)
        rule_total = time.time() - t_rb_start
        llm_total = 0.0
        n_rule = len(chunks)
        extraction_path = "rule_based"

    # Deduplicate fact_ids (can collide if chunk_idx * 1000 + span_idx wraps).
    seen: dict[str, int] = {}
    for f in facts:
        if f.fact_id in seen:
            seen[f.fact_id] += 1
            f.fact_id = f"{f.fact_id}_dup{seen[f.fact_id]}"
        else:
            seen[f.fact_id] = 0

    n_canonical = sum(
        1 for f in facts
        if f.canonical_form and f.raw_text and f.canonical_form != f.raw_text
    )
    n_weak = sum(
        1 for f in facts
        if f.self_containment_known and f.self_containment_score < MIN_SELF_CONTAINMENT
    )

    wall = time.time() - t0
    avg_llm = round(llm_total / len(llm_chunk_times), 2) if llm_chunk_times else 0.0
    logger.info(
        f"[extract] {doc_id}: {len(facts)} facts, {n_canonical} rewritten, "
        f"{n_weak} weak-containment, path={extraction_path}, {wall:.1f}s "
        f"[llm={len(llm_chunk_times)}chunks/{llm_total:.1f}s/{avg_llm}s/chunk "
        f"rule={n_rule}chunks/{rule_total:.1f}s]"
    )

    return ExtractResult(
        doc_id=doc_id,
        n_facts=len(facts),
        n_canonical=n_canonical,
        n_weak_containment=n_weak,
        extraction_path=extraction_path,
        wall_sec=wall,
        facts=facts,
        n_llm_chunks=len(llm_chunk_times),
        n_rule_chunks=n_rule,
        llm_total_sec=llm_total,
        rule_total_sec=rule_total,
        llm_chunk_sec_list=llm_chunk_times,
    )
