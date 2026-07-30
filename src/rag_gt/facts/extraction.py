"""Atomic fact extraction. Clause-verb detection before splitting on 'and'."""

from __future__ import annotations

import re
from typing import List

from rag_gt.chunking.strategies import _get_nlp
from rag_gt.core.config import load_config
from rag_gt.core.types import Document, Fact
from rag_gt.facts.quality import structural_quality
from rag_gt.facts.roles import detect_role

_cfg = load_config()
MIN_FACT_LEN = _cfg["fact_extraction"]["min_fact_length"]
MAX_FACT_LEN = _cfg["fact_extraction"]["max_fact_length"]
QUALITY_THRESH = _cfg["fact_extraction"]["quality_threshold"]

# Public id prefix so tests and consumers don't hard-code the format.
FACT_ID_PREFIX = "F"

_CLAUSE_SPLIT_RE = re.compile(
    r"\b(?P<conn>and|but|or|yet|so)\s+"
    r"(?:the|a|an|this|that|these|those|its|their|our|your|my|his|her)?\s*"
    r"\w+\s+"
    r"(?:is|are|was|were|has|have|had|shall|must|should|will|would|may|might|can|could|does|do|did)\b",
    re.IGNORECASE,
)


def _split_clauses(sentence: str) -> List[str]:
    if len(sentence) <= MAX_FACT_LEN:
        return [sentence]
    matches = list(_CLAUSE_SPLIT_RE.finditer(sentence))
    if not matches:
        return [sentence]
    parts: List[str] = []
    prev_end = 0
    for m in matches:
        part = sentence[prev_end : m.start()].strip()
        if part:
            parts.append(part)
        # Skip past the connector word so the next clause doesn't begin with "and "/"or "/etc.
        prev_end = m.end("conn") + 1  # +1 for the space the regex consumed after the conn word
        # Guard: ensure we don't run past the regex match start of the *next* word group.
        if prev_end > m.end():
            prev_end = m.start() + len(m.group("conn"))
    remainder = sentence[prev_end:].strip()
    if remainder:
        parts.append(remainder)
    return [p for p in parts if len(p) >= MIN_FACT_LEN]


def _truncate_clause(clause: str) -> str:
    """Cut a clause to MAX_FACT_LEN at the last word boundary, normalizing trailing punctuation."""
    if len(clause) <= MAX_FACT_LEN:
        return clause
    cut = clause[:MAX_FACT_LEN].rstrip()
    # Drop the partial word at the end.
    sp = cut.rfind(" ")
    if sp > 0:
        cut = cut[:sp]
    return cut.rstrip(".,;: ") + "."


def extract_candidate_facts(doc: Document, chunks: List[dict], llm=None) -> List[Fact]:
    """Extract atomic facts from chunks.

    The `llm` argument is currently unused; it is kept for backward compatibility
    with callers (tests + pipeline). A future LLM-assisted extraction path may
    consume it.
    """
    _ = llm  # accepted but unused; documented above.
    facts: List[Fact] = []
    fact_counter = 0
    nlp = _get_nlp()

    for chunk in chunks:
        sentences = chunk.get("sentences", [])
        if not sentences:
            spacy_doc = nlp(chunk["text"])
            sentences = [s.text.strip() for s in spacy_doc.sents if s.text.strip()]

        for sent in sentences:
            if len(sent.strip()) < MIN_FACT_LEN:
                continue
            clauses = _split_clauses(sent)
            for clause in clauses:
                clause = clause.strip()
                if len(clause) < MIN_FACT_LEN:
                    continue
                clause = _truncate_clause(clause)
                quality = structural_quality(clause)
                if quality < QUALITY_THRESH:
                    continue
                fact_counter += 1
                facts.append(
                    Fact(
                        fact_id=f"{doc.doc_id}_{FACT_ID_PREFIX}{fact_counter:04d}",
                        text=clause,
                        role=detect_role(clause),
                        weight=quality,
                    )
                )

    return facts
