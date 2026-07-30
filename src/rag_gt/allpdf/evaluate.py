"""Stage 10 — No-API deterministic RAG evaluation.

Takes the Q&A pairs produced by Stage 7 and evaluates them against a BM25
retriever built from the Stage-2 chunks. Runs entirely offline — no LLM API
calls, $0 marginal cost.

Metrics (per pair and corpus mean):
  fact_recall@k    fraction of required facts whose source chunk is retrieved in top-k
  hit@k            1 if ALL required chunks hit in top-k, else 0
  mrr              mean reciprocal rank of first required chunk
  coverage         fraction of fact canonical_form found as substring in retrieved chunks

These directly support the no-API evaluation novelty claim: arbitrary PDFs can
be evaluated on multi-hop reasoning quality without external judge calls.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger


# ---------------------------------------------------------------------------
# BM25 retriever (pure Python, no external dependency)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_bm25(chunks: List[dict]) -> Tuple[dict, dict, float, dict]:
    """Build BM25 index over chunks. Returns (inverted_index, doc_lens, avgdl, id_to_text)."""
    inverted: Dict[str, Dict[str, int]] = {}
    doc_lens: Dict[str, int] = {}
    id_to_text: Dict[str, str] = {}

    for chunk in chunks:
        cid = chunk.get("chunk_id", "")
        text = chunk.get("text", "")
        tokens = _tokenize(text)
        doc_lens[cid] = len(tokens)
        id_to_text[cid] = text
        for tok in tokens:
            inverted.setdefault(tok, {})
            inverted[tok][cid] = inverted[tok].get(cid, 0) + 1

    avgdl = sum(doc_lens.values()) / max(1, len(doc_lens))
    return inverted, doc_lens, avgdl, id_to_text


def _bm25_score(
    query_tokens: List[str],
    inverted: Dict[str, Dict[str, int]],
    doc_lens: Dict[str, int],
    avgdl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> Dict[str, float]:
    """Score all docs for a query; returns {chunk_id: score}."""
    import math

    N = len(doc_lens)
    scores: Dict[str, float] = {}
    for tok in set(query_tokens):
        df = len(inverted.get(tok, {}))
        if df == 0:
            continue
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        for cid, tf in inverted.get(tok, {}).items():
            dl = doc_lens.get(cid, avgdl)
            tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
            scores[cid] = scores.get(cid, 0.0) + idf * tf_norm
    return scores


def retrieve_bm25(
    query: str,
    inverted: Dict[str, Dict[str, int]],
    doc_lens: Dict[str, int],
    avgdl: float,
    top_k: int = 10,
) -> List[str]:
    """Return top-k chunk IDs ranked by BM25 score."""
    tokens = _tokenize(query)
    scores = _bm25_score(tokens, inverted, doc_lens, avgdl)
    ranked = sorted(scores, key=lambda c: scores[c], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class PairEval:
    pair_idx: int
    question: str
    depth: int
    n_required_facts: int
    retrieved_chunk_ids: List[str]

    # per-k metrics
    fact_recall_at_5: float = 0.0
    fact_recall_at_10: float = 0.0
    hit_at_1: int = 0
    hit_at_3: int = 0
    hit_at_5: int = 0
    hit_at_10: int = 0
    mrr: float = 0.0
    coverage: float = 0.0   # fraction of required facts with text found in top-10 chunks


@dataclass
class EvalResult:
    doc_id: str
    n_pairs: int
    mean_fact_recall_at_5: float
    mean_fact_recall_at_10: float
    mean_hit_at_1: float
    mean_hit_at_3: float
    mean_hit_at_5: float
    mean_hit_at_10: float
    mean_mrr: float
    mean_coverage: float
    per_pair: List[PairEval] = field(default_factory=list)
    wall_sec: float = 0.0

    def summary(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "n_pairs": self.n_pairs,
            "fact_recall@5": round(self.mean_fact_recall_at_5, 3),
            "fact_recall@10": round(self.mean_fact_recall_at_10, 3),
            "hit@1": round(self.mean_hit_at_1, 3),
            "hit@3": round(self.mean_hit_at_3, 3),
            "hit@5": round(self.mean_hit_at_5, 3),
            "hit@10": round(self.mean_hit_at_10, 3),
            "mrr": round(self.mean_mrr, 3),
            "coverage": round(self.mean_coverage, 3),
            "wall_sec": round(self.wall_sec, 2),
        }


def _fact_chunk_ids(fact: dict, chunks: List[dict]) -> List[str]:
    """Heuristic: return chunk IDs that contain the fact text as a substring."""
    text = (fact.get("canonical_form") or fact.get("text") or "").lower()
    if not text:
        return []
    text_tok = set(_tokenize(text))
    matched = []
    for c in chunks:
        c_tok = set(_tokenize(c.get("text", "")))
        overlap = len(text_tok & c_tok) / max(1, len(text_tok))
        if overlap >= 0.6:
            matched.append(c.get("chunk_id", ""))
    return [cid for cid in matched if cid]


def _text_covered(fact: dict, retrieved_texts: List[str]) -> bool:
    """True if >= 60% of the fact's tokens appear in any single retrieved chunk."""
    text = (fact.get("canonical_form") or fact.get("text") or "").lower()
    if not text:
        return False
    fact_tok = set(_tokenize(text))
    if not fact_tok:
        return False
    for ct in retrieved_texts:
        chunk_tok = set(_tokenize(ct))
        if len(fact_tok & chunk_tok) / len(fact_tok) >= 0.6:
            return True
    return False


def _mrr(required_chunk_ids: List[str], ranked: List[str]) -> float:
    for rank, cid in enumerate(ranked, start=1):
        if cid in set(required_chunk_ids):
            return 1.0 / rank
    return 0.0


def _hit(required_chunk_ids: List[str], ranked_k: List[str]) -> int:
    retrieved_set = set(ranked_k)
    return 1 if all(cid in retrieved_set for cid in required_chunk_ids) else 0


def _fact_recall(
    required_chunk_ids_per_fact: List[List[str]],
    retrieved_k: List[str],
) -> float:
    retrieved_set = set(retrieved_k)
    if not required_chunk_ids_per_fact:
        return 0.0
    hits = sum(
        1 for fact_chunks in required_chunk_ids_per_fact
        if any(cid in retrieved_set for cid in fact_chunks)
    )
    return hits / len(required_chunk_ids_per_fact)


def evaluate_pairs(
    doc_id: str,
    pairs: List[dict],
    chunks: List[dict],
    top_k: int = 10,
) -> EvalResult:
    """Evaluate Q&A pairs with BM25 retrieval over the Stage-2 chunks.

    Each pair must have: question, facts (list of {fact_id, text, canonical_form}).
    """
    t0 = time.time()
    if not pairs or not chunks:
        return EvalResult(doc_id=doc_id, n_pairs=0,
                          mean_fact_recall_at_5=0.0, mean_fact_recall_at_10=0.0,
                          mean_hit_at_1=0.0, mean_hit_at_3=0.0,
                          mean_hit_at_5=0.0, mean_hit_at_10=0.0,
                          mean_mrr=0.0, mean_coverage=0.0)

    inverted, doc_lens, avgdl, id_to_text = _build_bm25(chunks)
    chunk_ids_all = list(doc_lens.keys())

    pair_evals: List[PairEval] = []
    for idx, pair in enumerate(pairs):
        question = pair.get("question", "")
        facts = pair.get("facts", [])
        depth = pair.get("depth", len(facts))

        # Retrieve top-k chunks for this question.
        ranked_10 = retrieve_bm25(question, inverted, doc_lens, avgdl, top_k=10)
        ranked_5 = ranked_10[:5]
        ranked_3 = ranked_10[:3]
        ranked_1 = ranked_10[:1]

        # Required chunk IDs per fact (heuristic text overlap).
        req_per_fact = [_fact_chunk_ids(f, chunks) for f in facts]
        # All required chunk IDs (union).
        all_req = list({cid for fc in req_per_fact for cid in fc})

        retrieved_texts = [id_to_text[cid] for cid in ranked_10 if cid in id_to_text]
        n_covered = sum(1 for f in facts if _text_covered(f, retrieved_texts))
        coverage = n_covered / max(1, len(facts))

        pe = PairEval(
            pair_idx=idx,
            question=question,
            depth=depth,
            n_required_facts=len(facts),
            retrieved_chunk_ids=ranked_10,
            fact_recall_at_5=_fact_recall(req_per_fact, ranked_5),
            fact_recall_at_10=_fact_recall(req_per_fact, ranked_10),
            hit_at_1=_hit(all_req, ranked_1) if all_req else 0,
            hit_at_3=_hit(all_req, ranked_3) if all_req else 0,
            hit_at_5=_hit(all_req, ranked_5) if all_req else 0,
            hit_at_10=_hit(all_req, ranked_10) if all_req else 0,
            mrr=_mrr(all_req, ranked_10) if all_req else 0.0,
            coverage=coverage,
        )
        pair_evals.append(pe)

    def _mean(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    return EvalResult(
        doc_id=doc_id,
        n_pairs=len(pair_evals),
        mean_fact_recall_at_5=_mean([p.fact_recall_at_5 for p in pair_evals]),
        mean_fact_recall_at_10=_mean([p.fact_recall_at_10 for p in pair_evals]),
        mean_hit_at_1=_mean([float(p.hit_at_1) for p in pair_evals]),
        mean_hit_at_3=_mean([float(p.hit_at_3) for p in pair_evals]),
        mean_hit_at_5=_mean([float(p.hit_at_5) for p in pair_evals]),
        mean_hit_at_10=_mean([float(p.hit_at_10) for p in pair_evals]),
        mean_mrr=_mean([p.mrr for p in pair_evals]),
        mean_coverage=_mean([p.coverage for p in pair_evals]),
        per_pair=pair_evals,
        wall_sec=time.time() - t0,
    )


def evaluate_from_checkpoint(
    doc_id: str,
    pairs_path: str,
    chunks_path: str,
    out_path: Optional[str] = None,
) -> EvalResult:
    """Load Stage-7 pairs + Stage-2 chunks from checkpoint files and evaluate."""
    pairs = json.loads(Path(pairs_path).read_text(encoding="utf-8"))
    chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))

    logger.info(f"[eval] {doc_id}: {len(pairs)} pairs, {len(chunks)} chunks")
    result = evaluate_pairs(doc_id, pairs, chunks)
    logger.info(
        f"[eval] {doc_id}: fact_recall@5={result.mean_fact_recall_at_5:.3f} "
        f"hit@5={result.mean_hit_at_5:.3f} mrr={result.mean_mrr:.3f} "
        f"coverage={result.mean_coverage:.3f} ({result.wall_sec:.2f}s)"
    )

    if out_path:
        summary = result.summary()
        Path(out_path).write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    return result
