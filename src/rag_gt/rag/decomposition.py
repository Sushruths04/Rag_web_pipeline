"""Query-decomposition retriever for multi-hop (bridge) questions.

Inference-safe: uses ONLY the user question + an LLM to split it into standalone
sub-questions. It never reads gold facts, answers, or bridge labels. Each
sub-question is retrieved through a base retriever and the results are fused with
Reciprocal Rank Fusion, so a bridge question whose evidence lives on two pages can
surface BOTH hops instead of a single-shot retrieval that favours one.

Corpus-agnostic: no per-document logic. Any new PDF works unchanged.
"""

from __future__ import annotations

from typing import List, Tuple

from loguru import logger

from rag_gt.core.config import load_config


def _decompose_prompt(question: str) -> str:
    return f"""Split the question into the minimal set of standalone sub-questions
needed to answer it completely. Each sub-question must be self-contained and
retrievable on its own. If the question needs only one piece of evidence, return
just that single sub-question. Return at most 3. Do NOT answer them.

QUESTION: {question}

Return only JSON: {{"sub_questions": ["...", "..."]}}"""


class DecompositionRetriever:
    """Wrap a base retriever; retrieve per LLM-derived sub-question and RRF-fuse."""

    def __init__(self, base_retriever: object, llm: object, *, rrf_k: int | None = None, max_sub: int = 3) -> None:
        cfg = load_config()["multigold_evaluation"]
        self.base = base_retriever
        self.llm = llm
        self.rrf_k = int(cfg["rrf_k"]) if rrf_k is None else int(rrf_k)
        self.max_sub = int(max_sub)
        self._subq_cache: dict[str, list[str]] = {}

    def _sub_questions(self, query: str) -> list[str]:
        if query in self._subq_cache:
            return self._subq_cache[query]
        subs = [query]
        try:
            raw = self.llm.generate_json(
                _decompose_prompt(query), temperature=0.0, max_tokens=256
            )
            cand = [str(s).strip() for s in (raw.get("sub_questions") or []) if str(s).strip()]
            cand = [c for c in cand if len(c) >= 8][: self.max_sub]
            if cand:
                # Keep the full question too, so single-hop retrieval is never hurt.
                subs = list(dict.fromkeys(cand + [query]))
        except Exception as exc:  # decomposition is best-effort; fall back to the full query
            logger.debug(f"[decompose] fell back to full query: {type(exc).__name__}: {exc}")
        self._subq_cache[query] = subs
        return subs

    def prime_queries(self, queries: List[str]) -> None:
        all_sub: list[str] = []
        for q in queries:
            all_sub.extend(self._sub_questions(q))
        prime = getattr(self.base, "prime_queries", None)
        if callable(prime):
            prime(list(dict.fromkeys(list(queries) + all_sub)))

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        subs = self._sub_questions(query)
        per = max(top_k, 10)
        scores: dict[str, float] = {}
        for sq in subs:
            for rank, (cid, _score) in enumerate(self.base.retrieve(sq, top_k=per), start=1):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

    def get_chunk(self, chunk_id: str):
        return self.base.get_chunk(chunk_id)

    def get_chunk_text(self, chunk_id: str) -> str:
        return self.base.get_chunk_text(chunk_id)
