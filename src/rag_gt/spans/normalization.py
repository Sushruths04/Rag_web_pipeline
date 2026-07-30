"""Token-level fact-to-span mapping with trigram-indexed fuzzy fallback.

O(n log n), not O(n^2). `end_token` is EXCLUSIVE (Python slice semantics).
Chunk character ranges are treated as half-open [char_start, char_end).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import List, Tuple

from rapidfuzz import fuzz

from rag_gt.chunking.strategies import _get_nlp
from rag_gt.core.config import load_config
from rag_gt.core.types import Document, Fact, Span
from rag_gt.source_mapping import attach_source_units_to_range

_cfg = load_config()
FUZZ_THRESHOLD = _cfg["span_normalization"]["fuzzy_threshold"]
TAU_RECALL = _cfg["span_normalization"]["iou_tau_recall"]
TAU_PREC = _cfg["span_normalization"]["iou_tau_precision"]

# Stop words / very short anchors that should NOT be used as the unique
# trigram-indexed anchor in fuzzy matching. They produce too many candidates
# and dilute true matches.
_STOPWORD_ANCHORS = {
    "the", "a", "an", "and", "or", "but", "if", "is", "are", "was", "were",
    "of", "to", "in", "on", "for", "by", "with", "as", "at", "be", "this",
    "that", "these", "those", "it", "its", "their",
}


def tokenize_document(doc: Document) -> List[str]:
    """Return the document's tokens (text only). Use `tokenize_document_with_offsets`
    when you also need character offsets — `_token_to_char` no longer relies on
    re-finding tokens via `str.find`."""
    nlp = _get_nlp()
    spacy_doc = nlp(doc.text)
    return [t.text for t in spacy_doc]


def tokenize_document_with_offsets(doc: Document) -> Tuple[List[str], List[int]]:
    """Return (tokens, token_start_offsets) parsed once via spaCy."""
    nlp = _get_nlp()
    spacy_doc = nlp(doc.text)
    tokens: List[str] = []
    offsets: List[int] = []
    for t in spacy_doc:
        tokens.append(t.text)
        offsets.append(int(t.idx))
    return tokens, offsets


def _build_trigram_index(tokens: List[str]) -> dict[str, List[int]]:
    index: dict[str, List[int]] = defaultdict(list)
    for i, tok in enumerate(tokens):
        for tg in _token_trigrams(tok):
            index[tg].append(i)
    return index


def _token_trigrams(token: str) -> set[str]:
    t = token.lower()
    if len(t) < 3:
        return {t}
    return {t[i : i + 3] for i in range(len(t) - 2)}


def _exact_match_span(doc_tokens: List[str], fact_tokens: List[str]) -> Tuple[int, int]:
    if not fact_tokens:
        return (-1, -1)
    for i in range(len(doc_tokens) - len(fact_tokens) + 1):
        if doc_tokens[i : i + len(fact_tokens)] == fact_tokens:
            return (i, i + len(fact_tokens))
    return (-1, -1)


def _pick_anchor_indices(fact_tokens: List[str]) -> List[int]:
    """Pick up to 3 anchor token positions that aren't stopwords/short tokens."""
    candidates: List[int] = []
    for i, tok in enumerate(fact_tokens):
        low = tok.lower()
        if len(low) < 3:
            continue
        if low in _STOPWORD_ANCHORS:
            continue
        if not any(c.isalnum() for c in low):
            continue
        candidates.append(i)
        if len(candidates) >= 3:
            break
    if not candidates:
        candidates = [0]
    return candidates


def _fuzzy_match_span(
    doc_tokens: List[str], fact_tokens: List[str], trigram_index: dict[str, List[int]]
) -> Tuple[int, int]:
    if not fact_tokens:
        return (-1, -1)

    ft_len = len(fact_tokens)
    ft_lower_str = " ".join(t.lower() for t in fact_tokens)
    anchor_indices = _pick_anchor_indices(fact_tokens)

    candidates: set[int] = set()
    for ai in anchor_indices:
        for tg in _token_trigrams(fact_tokens[ai].lower()):
            if tg in trigram_index:
                # Each trigram hit at doc position p suggests fact_tokens[ai] aligns at p,
                # so the fact span starts near p - ai. Allow ±2 token slack.
                for p in trigram_index[tg]:
                    base = p - ai
                    for slack in (-2, -1, 0, 1, 2):
                        candidates.add(max(0, base + slack))

    if not candidates:
        return (-1, -1)

    best_score = 0
    best_start = -1
    best_end = -1
    # Sweep window lengths around ft_len to tolerate small insertions/deletions.
    length_offsets = (-2, -1, 0, 1, 2)
    for start in sorted(candidates):
        for delta in length_offsets:
            window_len = ft_len + delta
            if window_len < max(1, ft_len // 2):
                continue
            end = start + window_len
            if end > len(doc_tokens):
                continue
            window = doc_tokens[start:end]
            score = fuzz.ratio(" ".join(t.lower() for t in window), ft_lower_str)
            if score > best_score:
                best_score = score
                best_start = start
                best_end = end

    if best_score >= FUZZ_THRESHOLD and best_start >= 0:
        return (best_start, best_end)
    return (-1, -1)


def _token_iou_recall(
    doc_tokens: List[str], fact_tokens: List[str], start: int, end: int
) -> float:
    """Multiset IoU recall. Counts repeats — `the the X` does not let an unrelated
    span fake-cover the fact."""
    fact_counter = Counter(t.lower() for t in fact_tokens)
    if not fact_counter:
        return 0.0
    span_counter = Counter(t.lower() for t in doc_tokens[start:end])
    overlap = sum((fact_counter & span_counter).values())
    return overlap / sum(fact_counter.values())


def _token_iou_precision(
    doc_tokens: List[str], fact_tokens: List[str], start: int, end: int
) -> float:
    span_counter = Counter(t.lower() for t in doc_tokens[start:end])
    if not span_counter:
        return 0.0
    fact_counter = Counter(t.lower() for t in fact_tokens)
    overlap = sum((fact_counter & span_counter).values())
    return overlap / sum(span_counter.values())


def _token_to_char(
    doc_tokens: List[str],
    doc_text: str,
    token_idx: int,
    token_offsets: List[int] | None = None,
) -> int:
    """Return the character offset of `doc_tokens[token_idx]`.

    Prefers `token_offsets[token_idx]` (computed once via spaCy `Token.idx`).
    Falls back to a defensive substring scan that explicitly resolves the
    *target* token (the previous version returned the offset of the previous
    token, off-by-one).
    """
    if token_offsets is not None and 0 <= token_idx < len(token_offsets):
        return token_offsets[token_idx]
    if token_idx <= 0:
        return 0
    pos = 0
    for i in range(min(token_idx, len(doc_tokens))):
        idx = doc_text.find(doc_tokens[i], pos)
        if idx == -1:
            break
        pos = idx + len(doc_tokens[i])
    if token_idx < len(doc_tokens):
        idx = doc_text.find(doc_tokens[token_idx], pos)
        if idx != -1:
            return idx
    return pos


def find_fact_spans(
    doc: Document, doc_tokens: List[str], facts: List[Fact], chunks: List[dict]
) -> List[Fact]:
    """Attach `supporting_spans` to each fact by locating the fact text in the doc.

    `doc_tokens` may have been produced by `tokenize_document(doc)`. We re-tokenize
    here only to get authoritative token offsets via spaCy `Token.idx`.
    """
    trigram_index = _build_trigram_index(doc_tokens)
    nlp = _get_nlp()
    spacy_doc = nlp(doc.text)
    token_offsets = [int(t.idx) for t in spacy_doc]
    if len(token_offsets) != len(doc_tokens):
        # Defensive: fall back to find-based offsets only when the tokenizers diverge
        token_offsets = None  # type: ignore[assignment]

    for fact in facts:
        fact_doc = nlp(fact.text)
        fact_tokens = [t.text for t in fact_doc]

        start, end = _exact_match_span(doc_tokens, fact_tokens)
        if start == -1:
            start, end = _fuzzy_match_span(doc_tokens, fact_tokens, trigram_index)

        if start >= 0:
            recall = _token_iou_recall(doc_tokens, fact_tokens, start, end)
            precision = _token_iou_precision(doc_tokens, fact_tokens, start, end)
            if recall >= TAU_RECALL and precision >= TAU_PREC:
                char_pos = _token_to_char(doc_tokens, doc.text, start, token_offsets)
                if token_offsets is not None and end < len(token_offsets):
                    char_end = token_offsets[end]
                elif end > start and token_offsets is not None and end == len(token_offsets):
                    char_end = len(doc.text)
                else:
                    last_start = _token_to_char(
                        doc_tokens, doc.text, max(start, end - 1), token_offsets
                    )
                    last_token = doc_tokens[max(start, end - 1)] if doc_tokens else ""
                    char_end = min(len(doc.text), last_start + len(last_token))
                chunk_id = ""
                for c in chunks:
                    cs = c.get("char_start", 0)
                    ce = c.get("char_end", 0)
                    # Half-open [cs, ce). A fact at exactly `ce` belongs to the
                    # next chunk, not this one.
                    if cs <= char_pos < ce:
                        chunk_id = c["chunk_id"]
                        break
                source_meta = attach_source_units_to_range(doc, char_pos, char_end)

                fact.supporting_spans.append(
                    Span(
                        doc_id=doc.doc_id,
                        chunk_id=chunk_id,
                        start_token=start,
                        end_token=end,
                        char_start=char_pos,
                        char_end=char_end,
                        page_start=source_meta["page_start"],
                        page_end=source_meta["page_end"],
                        bboxes=source_meta["bboxes"],
                        block_ids=source_meta["block_ids"],
                        paragraph_ids=source_meta["paragraph_ids"],
                        source_text_sha1=source_meta["source_text_sha1"],
                        source_path=source_meta["source_path"],
                        source_sha1=source_meta["source_sha1"],
                        extractor=source_meta["extractor"],
                    )
                )

    return facts
