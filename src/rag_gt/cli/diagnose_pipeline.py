"""Standalone observability diagnostics for fact, pair, and gate attrition.

This command intentionally does not run the full question-generation pipeline.
It stops before expensive LLM question/answer generation and emits a dense JSON
audit that answers questions such as:

- Which raw facts were extracted?
- Which facts lost source spans or were removed by domain filters?
- Which fact pairs were even considered?
- Which pair gates or budgets removed pairs before TF-SFG classification?
- Which chains survived the cheap chain gates?
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

from rag_gt.chunking.strategies import chunk_document
from rag_gt.core.config import load_config
from rag_gt.facts.domain_filter import fact_domain_reject_reason, filter_fact_domain
from rag_gt.facts.extraction import extract_candidate_facts
from rag_gt.generation.multihop_sampler import enumerate_candidate_chains
from rag_gt.graph.pair_prefilter import PairPrefilterConfig, prefilter_pair
from rag_gt.graph.pair_scheduler import schedule_candidate_pairs
from rag_gt.ingestion import ingest_document
from rag_gt.observability.tracing import chain_snapshot, chunk_sample, fact_snapshot, make_chain_id
from rag_gt.pipeline.gt_pipeline import (
    _apply_depth_controls,
    _build_depth_distribution,
    _chain_char_gap,
    _chain_facts,
    _distinct_chain_chunks,
    _distinct_chain_pages,
    _facts_by_id,
    _filter_chains,
    _cache_key,
    _list_input_paths,
    _load_doc_cache,
    _normalized_fact_text,
    _stable_doc_seed,
)
from rag_gt.profiling.profiler import profile_document
from rag_gt.spans.normalization import find_fact_spans, tokenize_document
from rag_gt.validation.nli_check import nli_batch
from rag_gt.vectorstore.embedding import embed_facts
from rag_gt.vectorstore.faiss_index import FactIndex


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "and", "or", "but", "not",
    "with", "this", "that", "it", "its", "by", "from", "as", "into",
    "than", "then", "these", "those", "which", "while", "where", "when",
}


def _load_v16_config(cfg: dict) -> dict:
    path = cfg.get("v16", {}).get("config_path", "configs/v16.yaml")
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS}


def _bridge_tokens(a: Any, b: Any) -> int:
    return len(_tokens(getattr(a, "text", "")) & _tokens(getattr(b, "text", "")))


def _fact_page(fact: Any) -> int | None:
    for span in getattr(fact, "supporting_spans", []) or []:
        if getattr(span, "page_start", None) is not None:
            return int(span.page_start)
    return None


def _page_pair_key(a: Any, b: Any) -> tuple[int, int]:
    return (_fact_page(a) or -1, _fact_page(b) or -1)


def _fact_diag(fact: Any, *, status: str, reason: str = "") -> dict[str, Any]:
    snap = fact_snapshot(fact, include_text=True)
    text = str(getattr(fact, "text", "") or "")
    spans = list(getattr(fact, "supporting_spans", []) or [])
    span_dicts: list[dict[str, Any]] = []
    bbox_count = 0
    first_source_path = ""
    first_page = None
    for span in spans:
        sd = span.to_dict() if hasattr(span, "to_dict") else dict(span)
        span_dicts.append(sd)
        bbox_count += len(sd.get("bboxes", []) or [])
        if not first_source_path:
            first_source_path = str(sd.get("source_path", "") or "")
        if first_page is None and sd.get("page_start") is not None:
            first_page = int(sd.get("page_start"))
    pdf_page_link = ""
    if first_source_path and first_page is not None:
        # Browser-friendly local PDF link. Most browsers understand #page=N.
        pdf_page_link = (
            "file:///"
            + quote(str(Path(first_source_path).resolve()).replace("\\", "/"), safe="/:")
            + f"#page={first_page}"
        )
    snap.update(
        {
            "status": status,
            "reject_reason": reason,
            "char_len": len(text),
            "word_count": len(re.findall(r"\w+", text)),
            "domain_reject_reason": fact_domain_reject_reason(fact),
            "first_page": first_page,
            "source_path": first_source_path,
            "pdf_page_link": pdf_page_link,
            "bbox_count": bbox_count,
            "supporting_spans_full": span_dicts,
            "bbox_summary": [
                {
                    "page_no": b.get("page_no"),
                    "l": b.get("l"),
                    "t": b.get("t"),
                    "r": b.get("r"),
                    "b": b.get("b"),
                    "coord_origin": b.get("coord_origin"),
                }
                for sd in span_dicts
                for b in (sd.get("bboxes", []) or [])
            ],
        }
    )
    return snap


def _build_fact_index(facts: list[Any]) -> tuple[FactIndex | None, str]:
    if not facts:
        return None, "no_facts"
    try:
        embeddings = embed_facts(facts)
        index = FactIndex(dim=embeddings.shape[1])
        index.add(embeddings, [f.fact_id for f in facts])
        return index, "embedding_index"
    except Exception as exc:
        return None, f"embedding_failed:{type(exc).__name__}:{exc}"


def _lexical_pair_candidates(
    facts: list[Any],
    *,
    max_pairs: int,
) -> list[tuple[Any, Any, float]]:
    scored: list[tuple[float, Any, Any]] = []
    for i, a in enumerate(facts):
        ta = _tokens(getattr(a, "text", ""))
        if not ta:
            continue
        for j, b in enumerate(facts):
            if i == j:
                continue
            tb = _tokens(getattr(b, "text", ""))
            if not tb:
                continue
            score = len(ta & tb) / max(1, min(len(ta), len(tb)))
            if score > 0:
                scored.append((score, a, b))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(a, b, float(score)) for score, a, b in scored[:max_pairs]]


def _embedding_pair_candidates(
    facts: list[Any],
    index: FactIndex,
    *,
    top_k_neighbors: int,
    min_cosine_floor: float,
    max_pairs: int,
) -> list[tuple[Any, Any, float]]:
    by_id = {f.fact_id: f for f in facts}
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[Any, Any, float]] = []
    for fact in facts:
        try:
            emb = index.embedding_for(fact.fact_id)
        except KeyError:
            continue
        for nbr_id, score in index.search(emb, k=top_k_neighbors):
            if nbr_id == fact.fact_id or nbr_id not in by_id:
                continue
            if float(score) < min_cosine_floor:
                continue
            key = (fact.fact_id, nbr_id)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((fact, by_id[nbr_id], float(score)))
            if len(pairs) >= max_pairs:
                return pairs
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:max_pairs]


def _diagnose_pairs(
    facts: list[Any],
    index: FactIndex | None,
    *,
    cfg: dict,
    v16_cfg: dict,
    top_k_neighbors: int,
    min_pairs_for_analysis: int,
    max_pairs: int,
    use_v16_2_pair_prefilter: bool,
) -> dict[str, Any]:
    """Diagnose pair selection using the same scheduler as runtime TF-SFG."""
    tf_sfg_cfg = v16_cfg.get("tf_sfg", {}) or {}
    min_cosine = float(tf_sfg_cfg.get("min_cosine", 0.45))
    configured_max_candidate_pairs = int(tf_sfg_cfg.get("max_candidate_pairs", 800))
    max_candidate_pairs = min(configured_max_candidate_pairs, int(max_pairs))
    per_page_pair_cap = int(tf_sfg_cfg.get("per_page_pair_cap", 24))
    per_anchor_pair_cap = int(tf_sfg_cfg.get("per_anchor_pair_cap", 6))
    raw_candidate_pairs = int(tf_sfg_cfg.get("raw_candidate_pairs", 5000))
    canonical_source_order = bool(tf_sfg_cfg.get("canonical_source_order", True))
    pair_cfg_raw = (
        tf_sfg_cfg.get("pair_prefilter")
        or (v16_cfg.get("v16_2", {}) or {}).get("pair_prefilter", {})
    )
    pair_cfg = PairPrefilterConfig.from_dict(pair_cfg_raw)
    if not use_v16_2_pair_prefilter:
        pair_cfg.enabled = False
    redundancy_cfg = tf_sfg_cfg.get("pre_llm_redundancy", {}) or {}
    redundancy_enabled = bool(redundancy_cfg.get("enabled", False))
    redundancy_threshold = float(
        redundancy_cfg.get("bidirectional_entailment_threshold", 0.86)
    )
    single_fact_entailment_threshold = float(
        redundancy_cfg.get("single_fact_entailment_threshold", 1.01)
    )

    if index is not None:
        # Use the same shared scheduler as runtime TypedSFG
        pair_prefilter_fn = None
        if pair_cfg.enabled:
            pair_prefilter_fn = (
                lambda a, b, score: prefilter_pair(a, b, score, pair_cfg)
            )
        selected_triples, sched_report = schedule_candidate_pairs(
            facts,
            index,
            min_cosine=min_cosine,
            candidate_k=top_k_neighbors,
            raw_candidate_pairs=raw_candidate_pairs,
            max_pairs=max_candidate_pairs,
            per_page_pair_cap=per_page_pair_cap,
            per_anchor_pair_cap=per_anchor_pair_cap,
            pair_prefilter_fn=pair_prefilter_fn,
            source_neighbor_window=int(tf_sfg_cfg.get("source_neighbor_window", 0)),
            source_neighbor_max_page_gap=int(tf_sfg_cfg.get("source_neighbor_max_page_gap", 2)),
            canonical_source_order=canonical_source_order,
        )
        source = "embedding_scheduled"
    else:
        raw_lexical = _lexical_pair_candidates(facts, max_pairs=max_candidate_pairs)
        selected_triples = raw_lexical
        sched_report = None  # type: ignore[assignment]
        source = "lexical_fallback"

    rows: list[dict[str, Any]] = []
    hard_rejects: Counter[str] = Counter(
        (sched_report.to_dict().get("prefilter_stats", {}) if sched_report is not None else {})
        .get("hard_rejected", {})
    )
    redundant_pair_reasons: dict[str, str] = {}
    if redundancy_enabled and selected_triples:
        redundancy_pairs: list[tuple[str, str]] = []
        for a, b, _score in selected_triples:
            text_a = str(getattr(a, "canonical_form", "") or getattr(a, "text", "")).strip()
            text_b = str(getattr(b, "canonical_form", "") or getattr(b, "text", "")).strip()
            redundancy_pairs.extend([(text_a, text_b), (text_b, text_a)])
        redundancy_scores = nli_batch(redundancy_pairs)
        for idx, (a, b, _score) in enumerate(selected_triples):
            ab_score = redundancy_scores[idx * 2]
            ba_score = redundancy_scores[(idx * 2) + 1]
            if ab_score >= redundancy_threshold and ba_score >= redundancy_threshold:
                redundant_pair_reasons[f"{a.fact_id}__{b.fact_id}"] = (
                    "bidirectional_entailment"
                )
            elif max(ab_score, ba_score) >= single_fact_entailment_threshold:
                redundant_pair_reasons[f"{a.fact_id}__{b.fact_id}"] = (
                    "single_fact_entailment"
                )

    for idx, (a, b, score) in enumerate(selected_triples, start=1):
        if index is None:
            reject_reason, penalty = prefilter_pair(a, b, score, pair_cfg)
            adjusted = max(0.0, float(score) - float(penalty))
            status = "prefilter_rejected" if reject_reason else "selected_for_classification"
            if reject_reason:
                hard_rejects[reject_reason] += 1
        else:
            reject_reason, penalty = None, 0.0
            adjusted = float(score)
            status = "selected_for_classification"
        pair_id = f"{a.fact_id}__{b.fact_id}"
        if pair_id in redundant_pair_reasons:
            status = "pre_llm_redundant"
            reject_reason = redundant_pair_reasons[pair_id]
        row = {
            "pair_id": pair_id,
            "rank": idx,
            "src": a.fact_id,
            "dst": b.fact_id,
            "status": status,
            "reject_reason": reject_reason or "",
            "score": round(float(score), 5),
            "adjusted_score": round(adjusted, 5),
            "penalty": round(float(penalty), 5),
            "src_role": str(getattr(a, "role", "")),
            "dst_role": str(getattr(b, "role", "")),
            "src_page": _fact_page(a),
            "dst_page": _fact_page(b),
            "page_gap": (
                abs((_fact_page(a) or 0) - (_fact_page(b) or 0))
                if _fact_page(a) is not None and _fact_page(b) is not None
                else None
            ),
            "bridge_tokens": _bridge_tokens(a, b),
            "src_text": str(getattr(a, "text", "") or ""),
            "dst_text": str(getattr(b, "text", "") or ""),
        }
        rows.append(row)

    selected_ids = {r["pair_id"] for r in rows if r["status"] == "selected_for_classification"}
    status_counts = Counter(row["status"] for row in rows)
    sched_report_dict = sched_report.to_dict() if sched_report is not None else {}

    return {
        "source": source,
        "scheduler_report": sched_report_dict,
        "thresholds": {
            "embedding_min_cosine": min_cosine,
            "top_k_neighbors": top_k_neighbors,
            "max_pairs_for_analysis": max_pairs,
            "min_pairs_for_analysis": min_pairs_for_analysis,
            "pair_prefilter": pair_cfg.__dict__,
            "max_candidate_pairs": max_candidate_pairs,
            "configured_max_candidate_pairs": configured_max_candidate_pairs,
            "pre_llm_redundancy_enabled": redundancy_enabled,
            "bidirectional_entailment_threshold": redundancy_threshold,
            "single_fact_entailment_threshold": single_fact_entailment_threshold,
            "canonical_source_order": canonical_source_order,
            "per_page_pair_cap": per_page_pair_cap,
            "per_anchor_pair_cap": per_anchor_pair_cap,
        },
        "counts": {
            "raw_pairs": sched_report_dict.get("raw_candidate_count", len(rows)),
            "prefilter_rejected": sum(hard_rejects.values()),
            "budget_rejected": 0,
            "selected_for_classification": len(selected_ids),
            "pre_llm_redundant": len(redundant_pair_reasons),
            "pre_llm_redundant_by_reason": dict(Counter(redundant_pair_reasons.values())),
            "by_status": dict(status_counts),
            "prefilter_reject_by_reason": dict(hard_rejects),
        },
        "pairs": rows,
        "examples": {
            "selected": [r for r in rows if r["status"] == "selected_for_classification"][:25],
            "rejected": [r for r in rows if "rejected" in r["status"]][:25],
        },
    }


def _chain_reject_reason(
    chain: Any,
    by_id: dict[str, Any],
    *,
    question_mode: str,
    min_hops: int,
    max_hops: int,
    min_distinct_chunks: int,
    min_distinct_roles: int,
    min_distinct_pages: int,
    min_char_gap: int,
    chain_max_page_gap: int,
) -> str:
    if question_mode == "singlehop" and chain.depth != 1:
        return "question_mode_singlehop"
    if question_mode == "multihop" and chain.depth < 2:
        return "question_mode_multihop"
    if chain.depth < min_hops or chain.depth > max_hops:
        return "depth_out_of_range"
    cf = _chain_facts(chain, by_id)
    texts = [_normalized_fact_text(f) for f in cf]
    if len(set(texts)) < len(texts):
        return "duplicate_fact_text"
    if len(_distinct_chain_chunks(cf)) < min_distinct_chunks:
        return "insufficient_distinct_chunks"
    if len({str(f.role) for f in cf if f.role}) < min_distinct_roles:
        return "insufficient_distinct_roles"
    pages = _distinct_chain_pages(cf)
    if pages and len(pages) < min_distinct_pages:
        return "insufficient_distinct_pages"
    if min_char_gap > 0 and _chain_char_gap(cf) < min_char_gap:
        return "insufficient_char_gap"
    try:
        from rag_gt.generation.chain_quality import chain_quality_gate

        verdict = chain_quality_gate(cf, max_page_gap=chain_max_page_gap)
        if not verdict.accepted:
            return f"chain_quality:{verdict.reason}"
    except Exception as exc:
        return f"chain_quality_error:{type(exc).__name__}"
    return "kept"


def _diagnose_chains(
    facts: list[Any],
    index: FactIndex | None,
    *,
    cfg: dict,
    n_questions: int,
    question_mode: str,
    min_hops: int,
    max_hops: int,
    min_distinct_chunks: int,
    min_distinct_roles: int,
    min_distinct_pages: int,
    min_char_gap: int,
    doc_id: str,
) -> dict[str, Any]:
    target = max(1, n_questions)
    depth_dist = _apply_depth_controls(
        _build_depth_distribution(target, cfg),
        question_mode=question_mode,
        min_hops=min_hops,
        max_hops=max_hops,
    )
    min_cosine = float(cfg.get("multihop", {}).get("min_cosine", 0.55))
    role_bias = bool(cfg.get("multihop", {}).get("role_bias", True))
    chain_max_page_gap = int(cfg.get("chain_quality", {}).get("max_page_gap", 40) or 40)
    rng = random.Random(_stable_doc_seed(doc_id))
    chains = enumerate_candidate_chains(
        facts,
        index,
        n_chains_per_depth=depth_dist,
        min_cosine=min_cosine,
        role_bias=role_bias,
        rng=rng,
    )
    by_id = _facts_by_id(facts)
    filter_stats: dict[str, Any] = {}
    kept = _filter_chains(
        chains,
        by_id=by_id,
        question_mode=question_mode,
        min_hops=min_hops,
        max_hops=max_hops,
        min_distinct_chunks=min_distinct_chunks,
        min_distinct_roles=min_distinct_roles,
        min_distinct_pages=min_distinct_pages,
        min_char_gap=min_char_gap,
        chain_quality_enabled=bool(cfg.get("chain_quality", {}).get("enabled", True)),
        chain_max_page_gap=chain_max_page_gap,
        filter_stats=filter_stats,
    )
    kept_ids = {make_chain_id(c.fact_ids) for c in kept}
    rows = []
    for chain in chains:
        cid = make_chain_id(chain.fact_ids)
        reason = "kept" if cid in kept_ids else _chain_reject_reason(
            chain,
            by_id,
            question_mode=question_mode,
            min_hops=min_hops,
            max_hops=max_hops,
            min_distinct_chunks=min_distinct_chunks,
            min_distinct_roles=min_distinct_roles,
            min_distinct_pages=min_distinct_pages,
            min_char_gap=min_char_gap,
            chain_max_page_gap=chain_max_page_gap,
        )
        snap = chain_snapshot(chain, by_id)
        snap["status"] = "kept" if reason == "kept" else "rejected"
        snap["reject_reason"] = "" if reason == "kept" else reason
        rows.append(snap)

    depth_counts = Counter(int(r.get("depth", 0) or 0) for r in rows)
    kept_depth_counts = Counter(
        int(r.get("depth", 0) or 0) for r in rows if r["status"] == "kept"
    )
    rejected_depth_counts = Counter(
        int(r.get("depth", 0) or 0) for r in rows if r["status"] == "rejected"
    )
    kept_singlehop = sum(
        1 for r in rows if r["status"] == "kept" and int(r.get("depth", 0) or 0) == 1
    )
    kept_multihop = sum(
        1 for r in rows if r["status"] == "kept" and int(r.get("depth", 0) or 0) > 1
    )

    # --- Fix 7: distribution panels ---
    # accepted_by_page_bucket: group kept chains by median page // 10 bucket.
    # Catches "all accepted rows are front-matter" regressions.
    kept_rows = [r for r in rows if r["status"] == "kept"]
    page_bucket_counts: Counter[int] = Counter()
    for r in kept_rows:
        fids = r.get("fact_ids") or []
        pages = []
        for fid in (fids if isinstance(fids, list) else []):
            f = by_id.get(str(fid))
            if f is not None:
                pg = _fact_page(f)
                if pg is not None:
                    pages.append(pg)
        if pages:
            median_pg = sorted(pages)[len(pages) // 2]
            bucket = (median_pg // 10) * 10
            page_bucket_counts[bucket] += 1

    # accepted_by_section: use role_path as a coarse section proxy.
    section_counts: Counter[str] = Counter()
    for r in kept_rows:
        role_path = str(r.get("role_path", "") or "")
        section_counts[role_path or "unknown"] += 1

    # Per-reason sample chain_ids (≤3) for post-mortem without re-reading JSONL.
    reason_examples: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        reason = r.get("reject_reason") or ("kept" if r["status"] == "kept" else "unknown")
        if len(reason_examples[reason]) < 3:
            reason_examples[reason].append(str(r.get("chain_id", "")))

    return {
        "thresholds": {
            "depth_distribution": depth_dist,
            "min_cosine": min_cosine,
            "role_bias": role_bias,
            "question_mode": question_mode,
            "min_hops": min_hops,
            "max_hops": max_hops,
            "min_distinct_chunks": min_distinct_chunks,
            "min_distinct_roles": min_distinct_roles,
            "min_distinct_pages": min_distinct_pages,
            "min_char_gap": min_char_gap,
            "chain_max_page_gap": chain_max_page_gap,
        },
        "counts": {
            "sampled": len(chains),
            "kept": len(kept),
            "rejected": len(chains) - len(kept),
            "singlehop_kept": kept_singlehop,
            "multihop_kept": kept_multihop,
            "sampled_by_depth": dict(depth_counts),
            "kept_by_depth": dict(kept_depth_counts),
            "accepted_by_depth": dict(kept_depth_counts),   # alias for plan checkpoint naming
            "rejected_by_depth": dict(rejected_depth_counts),
            "reject_by_reason": dict(Counter(r["reject_reason"] or "kept" for r in rows)),
        },
        "distributions": {
            "accepted_by_depth": dict(kept_depth_counts),
            "accepted_by_page_bucket": {str(k): v for k, v in sorted(page_bucket_counts.items())},
            "accepted_by_section": dict(section_counts.most_common(20)),
            "drop_reason_examples": {k: v for k, v in reason_examples.items()},
        },
        "filter_stats": filter_stats,
        "chains": rows,
        "examples": {
            "kept": [r for r in rows if r["status"] == "kept"][:25],
            "rejected": [r for r in rows if r["status"] == "rejected"][:25],
        },
    }


def _diagnose_document(path: str, args: argparse.Namespace, cfg: dict, v16_cfg: dict) -> dict[str, Any]:
    t0 = time.time()
    cache_key = _cache_key(path, args.doc_type, args.chunk_size, args.chunk_overlap)
    cached = _load_doc_cache(cache_key)
    if cached is not None:
        doc, profile, cached_mapped_facts = cached
        chunks = chunk_document(doc, profile, args.chunk_size, args.chunk_overlap)
        raw_facts = list(cached_mapped_facts)
        span_kept = list(cached_mapped_facts)
        span_dropped: list[Any] = []
        fact_input_source = "doc_cache_mapped_facts"
    else:
        doc = ingest_document(path, doc_type=args.doc_type)
        profile = profile_document(
            doc,
            path=path,
            enable_fast_path=not args.disable_folder_heuristic,
        )
        chunks = chunk_document(doc, profile, args.chunk_size, args.chunk_overlap)
        raw_facts = extract_candidate_facts(doc, chunks, llm=None)
        doc_tokens = tokenize_document(doc)
        mapped_facts_all = find_fact_spans(doc, doc_tokens, raw_facts, chunks)
        span_kept = [f for f in mapped_facts_all if f.supporting_spans]
        span_dropped = [f for f in mapped_facts_all if not f.supporting_spans]
        fact_input_source = "fresh_extraction"
    domain_kept, domain_drops = filter_fact_domain(span_kept)
    domain_rejected = [
        f for f in span_kept if fact_domain_reject_reason(f) is not None
    ]

    index, index_status = _build_fact_index(domain_kept)
    pair_diag = _diagnose_pairs(
        domain_kept,
        index,
        cfg=cfg,
        v16_cfg=v16_cfg,
        top_k_neighbors=args.top_k_neighbors,
        min_pairs_for_analysis=args.min_pairs_for_analysis,
        max_pairs=args.max_pairs_per_doc,
        use_v16_2_pair_prefilter=not args.disable_pair_prefilter,
    )
    chain_diag = _diagnose_chains(
        domain_kept,
        index,
        cfg=cfg,
        n_questions=args.n_questions,
        question_mode=args.question_mode,
        min_hops=args.min_hops,
        max_hops=args.max_hops,
        min_distinct_chunks=args.min_distinct_chunks,
        min_distinct_roles=args.min_distinct_roles,
        min_distinct_pages=args.min_distinct_pages,
        min_char_gap=args.min_char_gap,
        doc_id=doc.doc_id,
    )

    return {
        "doc_id": doc.doc_id,
        "path": path,
        "elapsed_seconds": round(time.time() - t0, 3),
        "fact_input_source": fact_input_source,
        "ingestion": {
            "chars": len(doc.text),
            "source_backend": doc.source_backend,
            "source_units": len(doc.source_units),
        },
        "profile": profile,
        "chunks": {
            "count": len(chunks),
            "sample": chunk_sample(chunks, limit=10),
        },
        "facts": {
            "counts": {
                "raw_extracted": len(raw_facts),
                "span_mapped": len(span_kept),
                "span_dropped": len(span_dropped),
                "domain_kept": len(domain_kept),
                "domain_rejected": len(domain_rejected),
            },
            "domain_reject_by_reason": domain_drops,
            "raw_extracted_count_note": (
                "Cache contains source-mapped facts only; raw_extracted is the cached mapped input count."
                if fact_input_source == "doc_cache_mapped_facts"
                else "Fresh extraction count before source-span mapping."
            ),
            "raw_or_span_dropped": [
                _fact_diag(f, status="span_dropped", reason="missing_supporting_span")
                for f in span_dropped
            ],
            "domain_rejected": [
                _fact_diag(
                    f,
                    status="domain_rejected",
                    reason=str(fact_domain_reject_reason(f) or ""),
                )
                for f in domain_rejected
            ],
            "kept": [_fact_diag(f, status="kept") for f in domain_kept],
        },
        "index": {
            "status": index_status,
            "facts_indexed": len(domain_kept) if index is not None else 0,
        },
        "pairs": pair_diag,
        "chains": chain_diag,
        "summary": {
            "fact_survival_rate": (
                len(domain_kept) / len(raw_facts) if raw_facts else 0.0
            ),
            "pair_selection_rate": (
                pair_diag["counts"]["selected_for_classification"] / pair_diag["counts"]["raw_pairs"]
                if pair_diag["counts"]["raw_pairs"]
                else 0.0
            ),
            "chain_keep_rate": (
                chain_diag["counts"]["kept"] / chain_diag["counts"]["sampled"]
                if chain_diag["counts"]["sampled"]
                else 0.0
            ),
            "what_counts_mean": {
                "raw_extracted": "Facts initially extracted from chunks before source/domain gates.",
                "span_mapped": "Extracted facts that could be mapped back to source characters/chunks/pages/bboxes.",
                "domain_kept": "Facts that survived source-artifact/domain filtering and are eligible for embedding, pairing, and chains.",
                "raw_pairs": "Candidate directed fact pairs inspected for pair formation diagnostics.",
                "selected_for_classification": "Pairs that passed cheap pair gates and TF-SFG pair budget caps; these would be eligible for expensive edge classification.",
                "chains_kept": "Sampled fact chains that passed cheap chain gates before LLM question generation.",
            },
        },
    }


def _aggregate(report: dict[str, Any]) -> None:
    docs = report["documents"]
    totals = Counter()
    pair_status = Counter()
    fact_domain = Counter()
    chain_reasons = Counter()
    agg_page_bucket: Counter[str] = Counter()
    agg_section: Counter[str] = Counter()
    agg_depth: Counter[str] = Counter()
    for d in docs:
        fc = d["facts"]["counts"]
        for key, value in fc.items():
            totals[f"facts_{key}"] += int(value)
        pair_status.update(d["pairs"]["counts"].get("by_status", {}))
        fact_domain.update(d["facts"].get("domain_reject_by_reason", {}))
        chain_reasons.update(d["chains"]["counts"].get("reject_by_reason", {}))
        dists = d["chains"].get("distributions", {})
        for k, v in dists.get("accepted_by_page_bucket", {}).items():
            agg_page_bucket[k] += int(v)
        for k, v in dists.get("accepted_by_section", {}).items():
            agg_section[k] += int(v)
        for k, v in dists.get("accepted_by_depth", {}).items():
            agg_depth[k] += int(v)
    singlehop_kept = sum(int(d["chains"]["counts"].get("singlehop_kept", 0)) for d in docs)
    multihop_kept = sum(int(d["chains"]["counts"].get("multihop_kept", 0)) for d in docs)
    report["aggregate"] = {
        "documents": len(docs),
        "fact_counts": dict(totals),
        "pair_status": dict(pair_status),
        "fact_domain_reject_by_reason": dict(fact_domain),
        "chain_reject_by_reason": dict(chain_reasons),
        "chains": {
            "singlehop_kept": singlehop_kept,
            "multihop_kept": multihop_kept,
            "kept_total": singlehop_kept + multihop_kept,
        },
        "distributions": {
            "accepted_by_depth": dict(sorted(agg_depth.items())),
            "accepted_by_page_bucket": {k: v for k, v in sorted(agg_page_bucket.items(), key=lambda x: int(x[0]))},
            "accepted_by_section": dict(agg_section.most_common(20)),
        },
        "gate_flow": [
            {
                "gate": "fact_extraction",
                "input": None,
                "output": int(totals.get("facts_raw_extracted", 0)),
                "meaning": "Candidate facts extracted from chunk sentences/clauses.",
            },
            {
                "gate": "span_mapping",
                "input": int(totals.get("facts_raw_extracted", 0)),
                "output": int(totals.get("facts_span_mapped", 0)),
                "meaning": "Keeps facts that can be traced to source spans.",
            },
            {
                "gate": "domain_filter",
                "input": int(totals.get("facts_span_mapped", 0)),
                "output": int(totals.get("facts_domain_kept", 0)),
                "meaning": "Removes front matter, headings, references, captions, task prompts, and configured excluded pages.",
            },
            {
                "gate": "pair_prefilter_and_budget",
                "input": sum(pair_status.values()),
                "output": int(pair_status.get("selected_for_classification", 0)),
                "meaning": "Keeps fact pairs that pass cheap pair gates and pair-budget/page-pair caps.",
            },
            {
                "gate": "chain_filter",
                "input": sum(chain_reasons.values()),
                "output": int(chain_reasons.get("kept", 0)),
                "meaning": "Keeps sampled chains that pass depth/diversity/source-gap/chain-quality gates.",
            },
        ],
    }


def _write_html(report: dict[str, Any], output_path: Path) -> None:
    payload = json.dumps(report, ensure_ascii=False)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG-GT Fact/Pair Diagnostics</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f6f7f9;color:#1f2937}}
header{{position:sticky;top:0;background:#fff;border-bottom:1px solid #d9dee7;padding:14px 18px;z-index:2}}
h1{{font-size:19px;margin:0}} .sub{{font-size:12px;color:#667085;margin-top:3px}}
main{{padding:16px;max-width:1500px;margin:auto}}
.stats{{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:10px;margin-bottom:14px}}
.stat,.panel{{background:#fff;border:1px solid #d9dee7;border-radius:8px;box-shadow:0 1px 2px rgba(16,24,40,.06)}}
.stat{{padding:12px}}.stat b{{display:block;font-size:22px}}.stat span{{font-size:12px;color:#667085}}
.grid{{display:grid;grid-template-columns:320px 1fr;gap:14px;align-items:start}}
.panel{{padding:12px;margin-bottom:14px}}h2{{font-size:15px;margin:0 0 9px}}
input,select{{width:100%;padding:8px;border:1px solid #d9dee7;border-radius:6px;margin-bottom:8px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{padding:7px;border-bottom:1px solid #eef1f5;text-align:left;vertical-align:top}}
th{{color:#667085;background:#fbfcfe;position:sticky;top:50px}}tr:hover{{background:#f8fafc;cursor:pointer}}
.badge{{padding:2px 6px;border-radius:999px;font-size:11px;border:1px solid #d9dee7;display:inline-block}}
.kept,.selected_for_classification{{color:#137a52;background:#eaf8f1;border-color:#9bd9bd}}
.drop,.domain_rejected,.span_dropped,.prefilter_rejected,.budget_rejected_page_pair_cap,.budget_rejected_max_candidate_pairs,.rejected{{color:#b42318;background:#fff0ee;border-color:#f3a59d}}
.prefilter_passed{{color:#946200;background:#fff7db;border-color:#e6c76b}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#111827;color:#e5e7eb;border-radius:7px;padding:10px;max-height:560px;overflow:auto}}
.tabs{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}}button{{border:1px solid #cfd6e1;background:#fff;border-radius:6px;padding:7px 10px;cursor:pointer}}button.active{{background:#2563eb;color:white;border-color:#2563eb}}
.bar{{height:8px;background:#e9edf4;border-radius:999px;overflow:hidden}}.bar i{{display:block;height:100%;background:#2563eb}}
.help{{font-size:13px;line-height:1.45;color:#344054}}.help code{{background:#eef2f7;padding:1px 4px;border-radius:4px}}
.link{{color:#175cd3;text-decoration:none;font-weight:600}}
@media(max-width:900px){{.grid,.stats{{grid-template-columns:1fr}}th{{position:static}}}}
</style></head>
<body><header><h1>RAG-GT Fact/Pair Diagnostics</h1><div class="sub" id="sub"></div></header>
<main>
<section class="stats" id="stats"></section>
<section class="grid">
<aside>
<div class="panel"><h2>Filters</h2><select id="doc"></select><input id="q" placeholder="Search fact, pair, chain, reason, text"></div>
<div class="panel"><h2>Drop Reasons</h2><table id="reasons"><tbody></tbody></table></div>
<div class="panel"><h2>What This Means</h2><div class="help" id="help"></div></div>
<div class="panel"><h2>Selected Detail</h2><pre id="detail">Select a row.</pre></div>
</aside>
<section>
<div class="panel"><h2>Stage Tables</h2><div class="tabs" id="tabs"></div><table id="table"></table></div>
</section>
</section>
</main>
<script>
const REPORT={payload};
let tab='facts_kept';
function el(id){{return document.getElementById(id)}}
function esc(v){{return String(v??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}
function badge(s){{return `<span class="badge ${{esc(s)}}">${{esc(s)}}</span>`}}
function docs(){{return REPORT.documents||[]}}
function currentDoc(){{const id=el('doc').value; return docs().find(d=>d.doc_id===id)||docs()[0]}}
function allRows(){{const d=currentDoc(); if(!d)return[]; if(tab==='facts_kept')return d.facts.kept||[]; if(tab==='facts_dropped')return [...(d.facts.raw_or_span_dropped||[]),...(d.facts.domain_rejected||[])]; if(tab==='pairs')return d.pairs.pairs||[]; if(tab==='chains')return d.chains.chains||[]; return []}}
function columns(){{if(tab.startsWith('facts'))return['fact_id','status','reject_reason','role','weight','word_count','first_page','bbox_count','chunk_ids','pdf_page_link','text']; if(tab==='pairs')return['pair_id','status','reject_reason','score','adjusted_score','src_role','dst_role','src_page','dst_page','bridge_tokens','src_text','dst_text']; return['chain_id','status','reject_reason','depth','mean_cosine','role_path','fact_ids'];}}
function cell(c,v,r){{if(c==='status')return badge(v); if(c==='pdf_page_link'&&v)return `<a class="link" href="${{esc(v)}}" target="_blank">open page</a>`; return esc(Array.isArray(v)?v.join(', '):v)}}
function renderStats(){{const a=REPORT.aggregate||{{}}; const f=a.fact_counts||{{}}; const p=a.pair_status||{{}}; const c=a.chains||{{}}; const vals=[['Raw facts',f.facts_raw_extracted||0],['Domain kept',f.facts_domain_kept||0],['Domain rejected',f.facts_domain_rejected||0],['Pairs selected',p.selected_for_classification||0],['Single-hop kept',c.singlehop_kept||0],['Multi-hop kept',c.multihop_kept||0]]; el('stats').innerHTML=vals.map(x=>`<div class=stat><b>${{esc(x[1])}}</b><span>${{esc(x[0])}}</span></div>`).join('')}}
function renderDocSelect(){{el('doc').innerHTML=docs().map(d=>`<option>${{esc(d.doc_id)}}</option>`).join('')}}
function renderReasons(){{const d=currentDoc(); if(!d)return; const rows=[]; for(const [k,v] of Object.entries(d.facts.domain_reject_by_reason||{{}}))rows.push(['fact:'+k,v]); for(const [k,v] of Object.entries(d.pairs.counts.prefilter_reject_by_reason||{{}}))rows.push(['pair:'+k,v]); for(const [k,v] of Object.entries(d.chains.counts.reject_by_reason||{{}}))rows.push(['chain:'+k,v]); rows.sort((a,b)=>b[1]-a[1]); const max=Math.max(1,...rows.map(r=>r[1])); el('reasons').innerHTML='<tbody>'+rows.map(r=>`<tr><td>${{esc(r[0])}}</td><td>${{r[1]}}</td><td><div class=bar><i style="width:${{100*r[1]/max}}%"></i></div></td></tr>`).join('')+'</tbody>'}}
function renderTabs(){{const tabs=[['facts_kept','Kept facts'],['facts_dropped','Dropped facts'],['pairs','Pairs'],['chains','Chains']]; el('tabs').innerHTML=tabs.map(t=>`<button class="${{tab===t[0]?'active':''}}" onclick="tab='${{t[0]}}';renderAll()">${{t[1]}}</button>`).join('')}}
function renderTable(){{const q=el('q').value.toLowerCase(); const cols=columns(); const rows=allRows().filter(r=>!q||JSON.stringify(r).toLowerCase().includes(q)); el('table').innerHTML='<thead><tr>'+cols.map(c=>`<th>${{esc(c)}}</th>`).join('')+'</tr></thead><tbody>'+rows.map((r,i)=>`<tr data-i="${{i}}">`+cols.map(c=>`<td>${{cell(c,r[c],r)}}</td>`).join('')+'</tr>').join('')+'</tbody>'; [...el('table').querySelectorAll('tr[data-i]')].forEach((tr,i)=>tr.onclick=()=>{{const row=rows[i]; let extra=''; if(row.pdf_page_link)extra='\\n\\nOPEN PDF PAGE: '+row.pdf_page_link; el('detail').textContent=JSON.stringify(row,null,2)+extra}})}}
function renderSub(){{const d=currentDoc(); el('sub').textContent=d?`${{d.doc_id}} · facts ${{d.facts.counts.raw_extracted}} → ${{d.facts.counts.domain_kept}} · pairs selected ${{d.pairs.counts.selected_for_classification}}/${{d.pairs.counts.raw_pairs}} · chains kept ${{d.chains.counts.kept}}/${{d.chains.counts.sampled}}`:''}}
function renderHelp(){{const d=currentDoc(); if(!d)return; const p=d.pairs.counts; const c=d.chains.counts; el('help').innerHTML=`<p><b>Domain kept facts</b> are facts that survived source-span mapping and source/domain artifact filtering. These are eligible for embedding, pairing, and chain construction.</p><p><b>Pairs selected ${{p.selected_for_classification}}/${{p.raw_pairs}}</b> means the diagnostics inspected ${{p.raw_pairs}} directed candidate fact pairs and ${{p.selected_for_classification}} passed cheap pair gates and pair-budget/page-pair caps. These are eligible for expensive TF-SFG edge classification.</p><p><b>Chains kept ${{c.kept}}/${{c.sampled}}</b> means ${{c.sampled}} sampled fact chains were checked by depth/diversity/source-gap/chain-quality gates and ${{c.kept}} survived before LLM question generation. Kept single-hop=${{c.singlehop_kept||0}}, kept multi-hop=${{c.multihop_kept||0}}.</p><p><b>Answer/QA gates</b> happen only after LLM question/answer generation. Use the generation trace dashboard for QA-NLI, QRSG, answer generation, structure, quality, answer-NLI, and minimality rejection events.</p>`}}
function renderAll(){{renderStats();renderTabs();renderSub();renderReasons();renderHelp();renderTable()}}
renderDocSelect(); el('doc').oninput=renderAll; el('q').oninput=renderAll; renderAll();
</script></body></html>"""
    output_path.write_text(html, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config()
    v16_cfg = _load_v16_config(cfg)
    paths = _list_input_paths(
        args.input_dir,
        enable_docx=bool(cfg.get("ingestion", {}).get("enable_docx", False)),
    )
    if args.include:
        inc = args.include.lower()
        paths = [p for p in paths if inc in os.path.basename(p).lower()]
    if args.max_docs:
        paths = paths[: args.max_docs]
    if not paths:
        raise FileNotFoundError(f"No input documents found in {args.input_dir}")

    report: dict[str, Any] = {
        "schema_version": "rag_gt.pipeline_diagnostics.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_dir": args.input_dir,
        "config": {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "doc_type": args.doc_type,
            "question_mode": args.question_mode,
            "n_questions": args.n_questions,
            "min_hops": args.min_hops,
            "max_hops": args.max_hops,
            "min_distinct_chunks": args.min_distinct_chunks,
            "min_distinct_roles": args.min_distinct_roles,
            "min_distinct_pages": args.min_distinct_pages,
            "min_char_gap": args.min_char_gap,
            "top_k_neighbors": args.top_k_neighbors,
            "min_pairs_for_analysis": args.min_pairs_for_analysis,
            "max_pairs_per_doc": args.max_pairs_per_doc,
            "include": args.include,
        },
        "documents": [],
    }
    for path in paths:
        report["documents"].append(_diagnose_document(path, args, cfg, v16_cfg))
    _aggregate(report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag-gt-diagnose-pipeline",
        description="Generate a fact/pair/gate diagnostics JSON and HTML dashboard without full LLM generation.",
    )
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output", required=True, help="Diagnostics JSON output path")
    p.add_argument("--html", default=None, help="Dashboard HTML path; defaults to <output>.html")
    p.add_argument("--doc_type", default="UNKNOWN")
    p.add_argument("--n_questions", type=int, default=10)
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--chunk_overlap", type=int, default=64)
    p.add_argument("--max_docs", type=int, default=None)
    p.add_argument("--include", default="", help="Only diagnose input files whose basename contains this text.")
    p.add_argument("--disable_folder_heuristic", action="store_true")
    p.add_argument("--question_mode", choices=["singlehop", "multihop", "mixed"], default="mixed")
    p.add_argument("--min_hops", type=int, default=1)
    p.add_argument("--max_hops", type=int, default=3)
    p.add_argument("--min_distinct_chunks", type=int, default=1)
    p.add_argument("--min_distinct_roles", type=int, default=1)
    p.add_argument("--min_distinct_pages", type=int, default=1)
    p.add_argument("--min_char_gap", type=int, default=0)
    p.add_argument("--top_k_neighbors", type=int, default=12)
    p.add_argument("--min_pairs_for_analysis", type=int, default=10)
    p.add_argument("--max_pairs_per_doc", type=int, default=500)
    p.add_argument("--disable_pair_prefilter", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    report = run(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path = Path(args.html) if args.html else Path(f"{out}.html")
    _write_html(report, html_path)
    print(f"[diagnostics] wrote JSON: {out}")
    print(f"[diagnostics] wrote HTML: {html_path}")


if __name__ == "__main__":
    main()
