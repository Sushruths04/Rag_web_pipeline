"""Facts-level retrieval evaluation: RAG_GT-only, no answer needed.

Layered retrieval scoring against the curated ground-truth facts:

  L1  fact_recall          chunk_id of any supporting span was retrieved
  L2  text_recall          fact text lexically present in retrieved chunks
                           (rapidfuzz partial_ratio ≥ τ_L2)
  L3  text_recall_l3       fact text semantically present in retrieved chunks
                           (BGE cosine ≥ τ_L3)
  Strict (L1∧L2)  strict_recall
  Strict (L1∧L3)  strict_recall_l13   ← the headline replacement for RAGAS
                                        context_recall

  fact_precision           flat fact-grounded precision
  fact_precision_rw        rank-weighted precision (RAGAS-style) — replacement
                           for RAGAS context_precision

Plus IR metrics (Hit@k, MRR), per-role + per-difficulty breakdowns, and
bootstrap 95% confidence intervals on every corpus mean.

All measurements are deterministic and require zero LLM API calls.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from rag_gt.comparison.chunk_resolver import ChunkResolver, _normalized_key
from rag_gt.core.types import Fact, QuestionGT, RetrievalLog


# Layer thresholds. Documented defaults; override at the call site.
L2_PARTIAL_RATIO_THRESHOLD = 70.0   # rapidfuzz.partial_ratio in [0, 100]
L3_COSINE_THRESHOLD = 0.75          # BGE-normalised cosine in [-1, 1]


@dataclass
class FactHit:
    fact_id: str
    role: str
    text: str
    required_chunk_ids: List[str]            # chunk_ids that ground this fact
    retrieved: bool                          # L1: ≥1 required chunk was retrieved
    rank_of_first_hit: Optional[int] = None  # 1-indexed; None if not retrieved
    text_present: Optional[bool] = None      # L2: lexical text presence
    best_partial_ratio: Optional[float] = None
    strict_hit: Optional[bool] = None        # L1 ∧ L2
    text_present_l3: Optional[bool] = None   # L3: semantic text presence
    best_cosine: Optional[float] = None
    strict_hit_l13: Optional[bool] = None    # L1 ∧ L3 (the strict-replacement signal)


@dataclass
class PerQuestionRetrieval:
    q_id: str
    n_required_facts: int
    n_retrieved_chunks: int
    facts: List[FactHit] = field(default_factory=list)
    fact_recall: float = 0.0          # L1
    fact_precision: float = 0.0       # flat
    fact_f1: float = 0.0              # harmonic mean of L1 recall and flat precision
    fact_precision_rw: float = 0.0    # rank-weighted (RAGAS-style)
    joint_fact_recall: float = 0.0    # 1.0 only when every required fact is retrieved
    required_group_recall: float = 0.0
    multi_hop_success: float = 0.0
    overretrieval_penalty: float = 0.0
    span_recall: float = 0.0          # FSR
    span_precision: float = 0.0       # FSP
    hit_at_1: int = 0
    hit_at_3: int = 0
    hit_at_5: int = 0
    hit_at_10: int = 0
    mrr: float = 0.0
    missed_fact_ids: List[str] = field(default_factory=list)
    # L2 / L3 / strict aggregates (only populated when a ChunkResolver is supplied)
    text_recall: Optional[float] = None       # L2
    strict_recall: Optional[float] = None     # L1 ∧ L2
    text_recall_l3: Optional[float] = None    # L3
    strict_recall_l13: Optional[float] = None # L1 ∧ L3 — RAGAS context_recall analogue
    text_recall_any: float = 0.0              # L1 ∨ L2 ∨ L3 — "found anywhere"
    # Difficulty fields, copied through for grouping convenience
    reasoning_depth: int = 0
    semantic_distance: str = ""


@dataclass
class CorpusRetrieval:
    n_questions: int
    rows: List[PerQuestionRetrieval]
    means: Dict[str, float]
    medians: Dict[str, float]
    stdevs: Dict[str, float]
    cis: Dict[str, Tuple[float, float]]   # bootstrap 95% CI on each mean
    per_role_recall: Dict[str, float]
    fact_miss_rate: float
    questions_zero_recall: int
    questions_full_recall: int
    # Per-difficulty breakdowns: grouped_means[group_field][group_value][metric] -> mean
    grouped_means: Dict[str, Dict[Any, Dict[str, float]]] = field(default_factory=dict)


METRIC_KEYS = (
    "fact_recall", "fact_precision", "fact_f1", "fact_precision_rw",
    "joint_fact_recall", "required_group_recall", "multi_hop_success",
    "overretrieval_penalty",
    "span_recall", "span_precision",
    "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "mrr",
    "text_recall", "strict_recall",
    "text_recall_l3", "strict_recall_l13",
    "text_recall_any",   # L1 OR L2 OR L3 — most permissive "found anywhere"
)


def _required_chunk_ids(
    facts: List[Fact],
    resolver: Optional[ChunkResolver] = None,
) -> List[Set[str]]:
    required: List[Set[str]] = []
    for f in facts:
        ids: Set[str] = set()
        if resolver is not None:
            for s in f.supporting_spans:
                if s.char_start is None or s.char_end is None:
                    continue
                for cid in resolver.chunks_for_span(s):
                    ids.add(_normalized_key(cid))
        if not ids:
            ids = {_normalized_key(s.chunk_id) for s in f.supporting_spans if s.chunk_id}
        required.append(ids)
    return required


def evaluate_question(
    q_gt: QuestionGT,
    retrieval: RetrievalLog,
    resolver: Optional[ChunkResolver] = None,
    l2_threshold: float = L2_PARTIAL_RATIO_THRESHOLD,
    embedder: Optional[Any] = None,
    l3_threshold: float = L3_COSINE_THRESHOLD,
) -> PerQuestionRetrieval:
    """Compute layered per-question retrieval metrics.

    Args:
        q_gt: ground-truth question with `required_facts` populated.
        retrieval: the RAG system's retrieval log for this question.
        resolver: enables L2 (lexical text presence). Without it only L1
            metrics are computed.
        l2_threshold: rapidfuzz.partial_ratio cutoff for L2.
        embedder: a SentenceTransformer-like object exposing
            `.encode(texts, normalize_embeddings=True)`. When provided
            (in addition to `resolver`), enables L3 (semantic text presence)
            and strict_recall_l13.
        l3_threshold: cosine cutoff for L3 (BGE-normalised vectors).
    """
    facts = q_gt.required_facts
    if not facts:
        raise ValueError(
            f"GT question '{q_gt.q_id}' has no required_facts; regenerate GT."
        )

    retrieved = [_normalized_key(c) for c in retrieval.retrieved_chunk_ids]
    retrieved_set = set(retrieved)
    rank_of: Dict[str, int] = {}
    for i, cid in enumerate(retrieved, start=1):
        rank_of.setdefault(cid, i)

    fact_chunks = _required_chunk_ids(facts, resolver=resolver)

    # Resolve chunk text once for L2 and L3.
    retrieved_chunk_texts: List[str] = []
    if resolver is not None:
        for cid in retrieved:
            try:
                retrieved_chunk_texts.append(resolver.get(cid))
            except KeyError:
                pass

    # L3 setup: encode chunk texts once; encode each fact text on demand.
    chunk_vecs = None
    if embedder is not None and retrieved_chunk_texts:
        try:
            chunk_vecs = embedder.encode(
                retrieved_chunk_texts, normalize_embeddings=True, show_progress_bar=False
            )
        except TypeError:
            # Some encoders don't accept show_progress_bar
            chunk_vecs = embedder.encode(
                retrieved_chunk_texts, normalize_embeddings=True
            )

    # Rank-weighted precision: relevant_at_k indicator.
    # A retrieved chunk is "relevant" iff it appears in the union of supporting
    # chunk_ids for any required fact.
    supporting_ids_union: Set[str] = set()
    for req in fact_chunks:
        supporting_ids_union |= req
    rel_at_k = [1 if cid in supporting_ids_union else 0 for cid in retrieved]
    fact_precision_rw = _rank_weighted_precision(rel_at_k)

    fact_hits: List[FactHit] = []
    reciprocals: List[float] = []
    chunks_supporting_a_fact: Set[str] = set()
    n_retrieved_facts = 0
    span_total = 0
    span_matched = 0
    n_text_present = 0
    n_strict = 0
    n_text_present_l3 = 0
    n_strict_l13 = 0
    n_text_present_any = 0

    for f, req in zip(facts, fact_chunks):
        ranks = [rank_of[c] for c in req if c in rank_of]
        retrieved_flag = bool(ranks)
        first_rank = min(ranks) if ranks else None

        text_present: Optional[bool] = None
        best_ratio: Optional[float] = None
        strict_hit: Optional[bool] = None
        if resolver is not None:
            best_ratio = _best_partial_ratio(f.text, retrieved_chunk_texts)
            text_present = best_ratio >= l2_threshold
            strict_hit = bool(retrieved_flag and text_present)

        text_present_l3: Optional[bool] = None
        best_cosine: Optional[float] = None
        strict_hit_l13: Optional[bool] = None
        if embedder is not None and resolver is not None:
            best_cosine = _l3_max_cosine(f.text, chunk_vecs, embedder)
            text_present_l3 = best_cosine >= l3_threshold
            strict_hit_l13 = bool(retrieved_flag and text_present_l3)

        fact_hits.append(
            FactHit(
                fact_id=f.fact_id,
                role=f.role,
                text=f.text,
                required_chunk_ids=sorted(req),
                retrieved=retrieved_flag,
                rank_of_first_hit=first_rank,
                text_present=text_present,
                best_partial_ratio=best_ratio,
                strict_hit=strict_hit,
                text_present_l3=text_present_l3,
                best_cosine=best_cosine,
                strict_hit_l13=strict_hit_l13,
            )
        )
        if retrieved_flag:
            n_retrieved_facts += 1
            reciprocals.append(1.0 / first_rank)
            for c in req:
                if c in retrieved_set:
                    chunks_supporting_a_fact.add(c)
        span_total += len(req)
        span_matched += sum(1 for c in req if c in retrieved_set)
        if text_present:
            n_text_present += 1
        if strict_hit:
            n_strict += 1
        if text_present_l3:
            n_text_present_l3 += 1
        if strict_hit_l13:
            n_strict_l13 += 1
        # text_recall_any: fact found by ANY layer — chunk_id (L1), lexical (L2), or semantic (L3).
        # Designed to surface the verbatim-substring case where L3 cosine dilutes below threshold.
        if retrieved_flag or bool(text_present) or bool(text_present_l3):
            n_text_present_any += 1

    fact_recall = n_retrieved_facts / len(facts)
    fact_precision = (
        len(chunks_supporting_a_fact) / len(retrieved_set) if retrieved_set else 0.0
    )
    fact_f1 = _f1(fact_recall, fact_precision)
    joint_fact_recall = 1.0 if n_retrieved_facts == len(facts) else 0.0
    required_group_recall = _required_group_recall(q_gt, fact_hits)
    multi_hop_success = (
        joint_fact_recall
        if int(getattr(q_gt, "difficulty_reasoning_depth", 0) or 0) > 1
        else joint_fact_recall
    )
    overretrieval_penalty = (
        (len(retrieved_set) - len(chunks_supporting_a_fact)) / len(retrieved_set)
        if retrieved_set else 0.0
    )
    span_recall = span_matched / span_total if span_total else 1.0
    span_precision = (
        len(chunks_supporting_a_fact) / len(retrieved_set) if retrieved_set else float("nan")
    )

    def _hit_at(k: int) -> int:
        return int(any(r and r <= k for r in (h.rank_of_first_hit for h in fact_hits)))

    mrr = statistics.mean(reciprocals) if reciprocals else 0.0

    text_recall = (n_text_present / len(facts)) if resolver is not None else None
    strict_recall = (n_strict / len(facts)) if resolver is not None else None
    text_recall_l3 = (
        (n_text_present_l3 / len(facts)) if (embedder is not None and resolver is not None) else None
    )
    strict_recall_l13 = (
        (n_strict_l13 / len(facts)) if (embedder is not None and resolver is not None) else None
    )
    text_recall_any = n_text_present_any / len(facts)

    return PerQuestionRetrieval(
        q_id=q_gt.q_id,
        n_required_facts=len(facts),
        n_retrieved_chunks=len(retrieved_set),
        facts=fact_hits,
        fact_recall=fact_recall,
        fact_precision=fact_precision,
        fact_f1=fact_f1,
        fact_precision_rw=fact_precision_rw,
        joint_fact_recall=joint_fact_recall,
        required_group_recall=required_group_recall,
        multi_hop_success=multi_hop_success,
        overretrieval_penalty=overretrieval_penalty,
        span_recall=span_recall,
        span_precision=span_precision,
        hit_at_1=_hit_at(1),
        hit_at_3=_hit_at(3),
        hit_at_5=_hit_at(5),
        hit_at_10=_hit_at(10),
        mrr=mrr,
        missed_fact_ids=[h.fact_id for h in fact_hits if not h.retrieved],
        text_recall=text_recall,
        strict_recall=strict_recall,
        text_recall_l3=text_recall_l3,
        strict_recall_l13=strict_recall_l13,
        text_recall_any=text_recall_any,
        reasoning_depth=int(getattr(q_gt, "difficulty_reasoning_depth", 0) or 0),
        semantic_distance=str(getattr(q_gt, "difficulty_semantic_distance", "") or ""),
    )


def _best_partial_ratio(needle: str, haystacks: List[str]) -> float:
    if not needle or not haystacks:
        return 0.0
    from rapidfuzz import fuzz

    best = 0.0
    for h in haystacks:
        if not h:
            continue
        r = fuzz.partial_ratio(needle, h)
        if r > best:
            best = r
            if best >= 100.0:
                break
    return float(best)


def _l3_max_cosine(fact_text: str, chunk_vecs, embedder, aggregator: str = "max") -> float:
    """Return the L3 cosine score between fact_text and the retrieved chunks.

    aggregator:
      - "max": maximum cosine (default — strictest single-chunk evidence)
      - "top3_mean": mean of the top-3 cosines (smoother, less spiky)
    chunk_vecs is assumed L2-normalised.
    """
    if chunk_vecs is None or len(chunk_vecs) == 0 or not fact_text:
        return 0.0
    try:
        fv = embedder.encode([fact_text], normalize_embeddings=True, show_progress_bar=False)[0]
    except TypeError:
        fv = embedder.encode([fact_text], normalize_embeddings=True)[0]
    # Both fv and chunk_vecs are unit-norm → dot product is cosine.
    sims = chunk_vecs @ fv
    if aggregator == "top3_mean":
        import numpy as _np
        k = min(3, len(sims))
        top = _np.partition(sims, -k)[-k:]
        return float(top.mean())
    return float(sims.max())


def _rank_weighted_precision(rel_at_k: Sequence[int]) -> float:
    """RAGAS-style rank-weighted precision.

    Given a binary relevance vector over the retrieved-chunk ranking,
    compute ``Σ_k (Precision@k · v_k) / Σ_k v_k``. Returns 0.0 if no
    retrieved chunk is relevant; returns 0.0 on empty input.
    """
    if not rel_at_k:
        return 0.0
    total_rel = sum(rel_at_k)
    if total_rel == 0:
        return 0.0
    cum = 0
    weighted_sum = 0.0
    for k, v in enumerate(rel_at_k, start=1):
        cum += v
        if v:
            weighted_sum += (cum / k)
    return weighted_sum / total_rel


def _f1(recall: float, precision: float) -> float:
    denom = recall + precision
    if denom <= 0:
        return 0.0
    return 2 * recall * precision / denom


def _required_group_recall(q_gt: QuestionGT, fact_hits: Sequence[FactHit]) -> float:
    groups = getattr(q_gt, "required_fact_groups", None) or []
    if not groups:
        groups = [list(getattr(q_gt, "required_fact_ids", []) or [])]
    if not groups:
        return 1.0
    hit_by_fact = {h.fact_id: h.retrieved for h in fact_hits}
    recovered = 0
    total = 0
    for group in groups:
        fact_ids = [str(fid) for fid in group if str(fid)]
        if not fact_ids:
            continue
        total += 1
        if all(hit_by_fact.get(fid, False) for fid in fact_ids):
            recovered += 1
    return recovered / total if total else 1.0


def _bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI on the mean. Returns (lo, hi)."""
    cleaned = [float(v) for v in values if isinstance(v, (int, float)) and v == v]
    if len(cleaned) < 2:
        if cleaned:
            return (cleaned[0], cleaned[0])
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(cleaned)
    means: List[float] = []
    for _ in range(n_resamples):
        s = 0.0
        for _ in range(n):
            s += cleaned[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo_idx = int(n_resamples * (alpha / 2))
    hi_idx = int(n_resamples * (1 - alpha / 2)) - 1
    hi_idx = max(hi_idx, 0)
    return (means[lo_idx], means[hi_idx])


def evaluate_corpus(
    q_map: Dict[str, QuestionGT],
    ret_logs: Dict[str, RetrievalLog],
    resolver: Optional[ChunkResolver] = None,
    l2_threshold: float = L2_PARTIAL_RATIO_THRESHOLD,
    embedder: Optional[Any] = None,
    l3_threshold: float = L3_COSINE_THRESHOLD,
    bootstrap_resamples: int = 1000,
) -> CorpusRetrieval:
    matched = sorted(q for q in q_map if q in ret_logs)
    rows = [
        evaluate_question(
            q_map[qid], ret_logs[qid],
            resolver=resolver, l2_threshold=l2_threshold,
            embedder=embedder, l3_threshold=l3_threshold,
        )
        for qid in matched
    ]

    if not rows:
        return CorpusRetrieval(
            n_questions=0, rows=[], means={}, medians={}, stdevs={}, cis={},
            per_role_recall={}, fact_miss_rate=float("nan"),
            questions_zero_recall=0, questions_full_recall=0, grouped_means={},
        )

    means, medians, stdevs, cis = {}, {}, {}, {}
    for k in METRIC_KEYS:
        vals = [getattr(r, k) for r in rows]
        cleaned = [v for v in vals if isinstance(v, (int, float)) and v == v]
        if cleaned:
            means[k] = statistics.mean(cleaned)
            medians[k] = statistics.median(cleaned)
            stdevs[k] = statistics.pstdev(cleaned) if len(cleaned) > 1 else 0.0
            cis[k] = _bootstrap_ci(cleaned, n_resamples=bootstrap_resamples)
        else:
            means[k] = float("nan")
            medians[k] = float("nan")
            stdevs[k] = float("nan")
            cis[k] = (float("nan"), float("nan"))

    # Per-role recall.
    role_total: Dict[str, int] = {}
    role_hit: Dict[str, int] = {}
    fact_total = 0
    fact_misses = 0
    for r in rows:
        for h in r.facts:
            role_total[h.role] = role_total.get(h.role, 0) + 1
            if h.retrieved:
                role_hit[h.role] = role_hit.get(h.role, 0) + 1
            else:
                fact_misses += 1
            fact_total += 1
    per_role_recall = {
        role: role_hit.get(role, 0) / role_total[role] for role in sorted(role_total)
    }

    # Per-difficulty grouping.
    grouped_means = _group_means(rows, METRIC_KEYS)

    return CorpusRetrieval(
        n_questions=len(rows),
        rows=rows,
        means=means,
        medians=medians,
        stdevs=stdevs,
        cis=cis,
        per_role_recall=per_role_recall,
        fact_miss_rate=fact_misses / fact_total if fact_total else float("nan"),
        questions_zero_recall=sum(1 for r in rows if r.fact_recall == 0.0),
        questions_full_recall=sum(1 for r in rows if r.fact_recall == 1.0),
        grouped_means=grouped_means,
    )


def _group_means(
    rows: List[PerQuestionRetrieval], metric_keys: Sequence[str]
) -> Dict[str, Dict[Any, Dict[str, float]]]:
    """Group per-question metrics by reasoning_depth and semantic_distance."""
    groups: Dict[str, Dict[Any, List[PerQuestionRetrieval]]] = {
        "reasoning_depth": {},
        "semantic_distance": {},
    }
    for r in rows:
        if r.reasoning_depth:
            groups["reasoning_depth"].setdefault(r.reasoning_depth, []).append(r)
        if r.semantic_distance:
            groups["semantic_distance"].setdefault(r.semantic_distance, []).append(r)

    out: Dict[str, Dict[Any, Dict[str, float]]] = {}
    for field_name, by_value in groups.items():
        out[field_name] = {}
        for value, group_rows in by_value.items():
            out[field_name][value] = {"n": len(group_rows)}
            for k in metric_keys:
                vals = [getattr(r, k) for r in group_rows]
                cleaned = [v for v in vals if isinstance(v, (int, float)) and v == v]
                out[field_name][value][k] = (
                    statistics.mean(cleaned) if cleaned else float("nan")
                )
    return out
