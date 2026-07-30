"""Provenance-Anchored Synthetic Distractors (PASD) — v16 P3.

For each chain fact, mine 2–3 distractor spans from the same corpus satisfying:
  - cosine ≥ min_cosine to the chain fact (semi-relevant — will surface in retrieval)
  - NLI non-entail to the gold answer (distractor does not contain the answer)
  - Different page or section than the support span (forces retriever to disambiguate)
  - Distinct fact_id from all chain facts

Each distractor carries full provenance (page, bbox, char_span) so evaluators
can inject it into retrieval during the hard-track evaluation.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Set

from loguru import logger

from rag_gt.core.types import Fact
from rag_gt.validation.nli_check import nli_entailment
from rag_gt.vectorstore.faiss_index import FactIndex

_CANDIDATE_K = 25          # neighbours checked per chain fact
_DISTRACTOR_ID_PREFIX = "d"


def _fact_pages(fact: Fact) -> Set[int]:
    pages: Set[int] = set()
    for span in fact.supporting_spans:
        if span.page_start is not None:
            pages.add(int(span.page_start))
        if span.page_end is not None:
            pages.add(int(span.page_end))
    return pages


def _fact_primary_bbox(fact: Fact) -> Optional[list]:
    for span in fact.supporting_spans:
        if span.bboxes:
            b = span.bboxes[0]
            return [b.l, b.t, b.r, b.b]
    return None


def _fact_char_span(fact: Fact) -> Optional[list]:
    for span in fact.supporting_spans:
        if span.char_start is not None and span.char_end is not None:
            return [int(span.char_start), int(span.char_end)]
    return None


def _fact_text(fact: Fact) -> str:
    return (fact.canonical_form.strip() or fact.text.strip())


def mine_distractors(
    gold_answer: str,
    chain_facts: List[Fact],
    all_facts_by_id: Dict[str, Fact],
    index: FactIndex,
    *,
    per_fact: int = 2,
    min_cosine: float = 0.55,
    max_nli_to_answer: float = 0.30,
    candidate_k: int = _CANDIDATE_K,
    distractor_id_offset: int = 0,
) -> List[dict]:
    """Mine provenance-anchored distractors for a chain.

    Args:
        gold_answer: accepted gold answer for the question.
        chain_facts: the SFUs forming the chain.
        all_facts_by_id: all facts in the document, keyed by fact_id.
        index: the FAISS fact index for the document.
        per_fact: target number of distractors per chain fact.
        min_cosine: cosine lower bound for candidate distractors.
        max_nli_to_answer: NLI upper bound (distractor must NOT entail the answer).
        candidate_k: top-K neighbours checked per chain fact.
        distractor_id_offset: base for distractor_id counter (for uniqueness across rows).

    Returns:
        List of distractor dicts in the QuestionGT.distractor_spans schema.
    """
    chain_ids: Set[str] = {f.fact_id for f in chain_facts}
    chain_pages: Set[int] = set()
    for f in chain_facts:
        chain_pages.update(_fact_pages(f))

    out: List[dict] = []
    seen_distractor_ids: Set[str] = set()
    counter = itertools.count(distractor_id_offset + 1)

    for source_fact in chain_facts:
        try:
            emb = index.embedding_for(source_fact.fact_id)
        except KeyError:
            continue

        neighbours = index.search(emb, k=candidate_k)
        collected = 0

        for nbr_id, cosine in neighbours:
            if collected >= per_fact:
                break
            if nbr_id in chain_ids:
                continue
            if nbr_id in seen_distractor_ids:
                continue
            if nbr_id not in all_facts_by_id:
                continue
            if cosine < min_cosine:
                continue

            nbr_fact = all_facts_by_id[nbr_id]
            nbr_pages = _fact_pages(nbr_fact)

            # Require different page when page metadata is available for both.
            if chain_pages and nbr_pages and not nbr_pages.isdisjoint(chain_pages):
                continue

            # NLI filter: distractor must NOT entail the gold answer.
            nbr_text = _fact_text(nbr_fact)
            try:
                nli_score = nli_entailment(nbr_text, gold_answer)
            except Exception as e:
                logger.debug(f"[PASD] NLI error for {nbr_id}: {e}")
                nli_score = 0.0

            if nli_score >= max_nli_to_answer:
                continue

            did = f"{_DISTRACTOR_ID_PREFIX}{next(counter):05d}"
            seen_distractor_ids.add(nbr_id)

            page = next(iter(nbr_pages), None)
            bbox = _fact_primary_bbox(nbr_fact)
            char_span = _fact_char_span(nbr_fact)

            out.append({
                "distractor_id": did,
                "doc_id": (
                    nbr_fact.supporting_spans[0].doc_id
                    if nbr_fact.supporting_spans else ""
                ),
                "page": page,
                "bbox": bbox,
                "char_span": char_span,
                "source_fact_id": source_fact.fact_id,
                "cosine_to_support": round(float(cosine), 4),
                "nli_to_question": round(float(nli_score), 4),
            })
            collected += 1

    return out
