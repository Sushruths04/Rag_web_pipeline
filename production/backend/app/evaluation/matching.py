"""LLM-free GT matching: token-containment scoring of retrieved chunks.

Same rule as rag_gt.rag.matcher (>=60% of a gold unit's tokens present in a
single retrieved chunk), reimplemented here without rag_gt config coupling so
it is pure and unit-testable. Gold units are the GT pair's answer_clauses
texts (each clause is one evidence unit, typically one fact); pairs without
clauses fall back to the answer text as a single unit.

Chunk-agnostic by construction: no chunk ids are compared, only text
containment — so any chunking strategy can be scored against the same GT.
"""
from __future__ import annotations

import re

OVERLAP_THRESHOLD = 0.60

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def unit_hit(unit_text: str, chunk_texts: list[str]) -> bool:
    """True when >=60% of the unit's tokens appear in ONE retrieved chunk."""
    unit_tokens = tokenize(unit_text)
    if not unit_tokens:
        return False
    needed = len(unit_tokens) * OVERLAP_THRESHOLD
    for chunk in chunk_texts:
        chunk_tokens = set(tokenize(chunk))
        covered = sum(1 for t in unit_tokens if t in chunk_tokens)
        if covered >= needed:
            return True
    return False


def precision_rw(rel_at_k: list[int]) -> float:
    """RAGAS-style rank-weighted precision: Σ_k (Precision@k · rel_k) / Σ_k rel_k.

    Same formula as rag_gt.comparison.retrieval_metrics._rank_weighted_precision
    (the v10 fact_precision_rw / RAGAS context_precision comparable).
    Unlike raw precision it does not punish a question for having fewer gold
    units than retrieved chunks: 1 unit ranked #1 scores 1.0, not 1/k.
    """
    if not rel_at_k:
        return 0.0
    total_rel = sum(rel_at_k)
    if total_rel == 0:
        return 0.0
    cum = 0
    weighted = 0.0
    for k, v in enumerate(rel_at_k, start=1):
        cum += v
        if v:
            weighted += cum / k
    return weighted / total_rel


def _gold_units(pair: dict) -> list[str]:
    clauses = pair.get("answer_clauses") or []
    units = [c["text"] for c in clauses if isinstance(c, dict) and c.get("text")]
    if units:
        return units
    answer = (pair.get("answer") or "").strip()
    return [answer] if answer else []


def score_pair(pair: dict, retrieved_texts: list[str]) -> dict:
    """Score one GT pair against the retrieved chunk texts."""
    units = _gold_units(pair)
    hits = [unit_hit(u, retrieved_texts) for u in units]
    recall = sum(hits) / len(units) if units else 0.0
    rel_at_k = [
        1 if any(unit_hit(u, [c]) for u in units) else 0 for c in retrieved_texts
    ]
    relevant = sum(rel_at_k)
    precision = relevant / len(retrieved_texts) if retrieved_texts else 0.0
    return {
        "qa_id": pair.get("qa_id"),
        "hop_type": pair.get("hop_type", "single"),
        "recall": recall,
        "precision": precision,
        "precision_rw": precision_rw(rel_at_k),
        "rel_at_k": rel_at_k,
        "hit": bool(units) and all(hits),
        "n_units": len(units),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(rows: list[dict]) -> dict:
    recall = _mean([r["recall"] for r in rows])
    precision = _mean([r["precision"] for r in rows])
    prec_rw = _mean([r["precision_rw"] for r in rows])
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    f1_rw = 2 * recall * prec_rw / (recall + prec_rw) if (recall + prec_rw) > 0 else 0.0
    by_hop: dict[str, dict] = {}
    for hop in sorted({r["hop_type"] for r in rows}):
        sub = [r for r in rows if r["hop_type"] == hop]
        by_hop[hop] = {
            "n": len(sub),
            "recall": _mean([r["recall"] for r in sub]),
            "precision": _mean([r["precision"] for r in sub]),
            "precision_rw": _mean([r["precision_rw"] for r in sub]),
            "hit_rate": _mean([1.0 if r["hit"] else 0.0 for r in sub]),
        }
    return {
        "n": len(rows),
        "recall": recall,
        "precision": precision,
        "precision_rw": prec_rw,
        "f1": f1,
        "f1_rw": f1_rw,
        "hit_rate": _mean([1.0 if r["hit"] else 0.0 for r in rows]),
        "by_hop": by_hop,
    }


def pick_winner(rows: list[dict]) -> dict:
    """Leaderboard winner: highest f1_rw; on ties the smaller top_k wins
    (same tie direction as the old ``(f1, -top_k)`` key)."""
    return max(rows, key=lambda r: (r["f1_rw"], -r["top_k"]))
