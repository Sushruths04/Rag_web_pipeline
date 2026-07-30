"""Three chunking strategies. `sentences` / `tokens` are List[str], never spaCy objects.

All chunks carry `(char_start, char_end)` offsets into the *original* `doc.text`.
Offsets are computed from spaCy `Token.idx` and accumulated section bases — never
recovered by `str.find`, which mis-aligns when phrases recur (TOC vs body).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List

import spacy

from rag_gt.core.types import Document
from rag_gt.source_mapping import annotate_chunk_with_source


@lru_cache(maxsize=1)
def _get_nlp():
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])
    except OSError:
        nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def chunk_document(
    doc: Document, profile: dict, chunk_size: int = 512, chunk_overlap: int = 64
) -> List[dict]:
    dt = profile.get("doc_type", "UNKNOWN")
    if dt == "ISO_STANDARD":
        return _with_current_doc(doc, _structure_semantic, doc, chunk_size, chunk_overlap)
    if dt in ("KT_DOC", "TEXTBOOK"):
        return _with_current_doc(doc, _heading_semantic, doc, chunk_size, chunk_overlap)
    return _with_current_doc(doc, _recursive_semantic, doc, chunk_size, chunk_overlap)


def _structure_semantic(doc, chunk_size, overlap):
    # Allow tabs as well as spaces between clause number and title.
    clause_re = re.compile(r"(?m)^(\d+(?:\.\d+)*\.?[ \t]{1,4}[A-Z])")
    splits = [m.start() for m in clause_re.finditer(doc.text)]
    if not splits:
        return _recursive_semantic(doc, chunk_size, overlap)
    splits.append(len(doc.text))
    chunks = []
    for i in range(len(splits) - 1):
        section = doc.text[splits[i] : splits[i + 1]]
        chunks.extend(
            _section_to_chunks(section, doc, chunk_size, overlap, len(chunks), splits[i])
        )
    return chunks


def _heading_semantic(doc, chunk_size, overlap):
    # Use [A-Z\t ] (no \s) so the all-caps heading match cannot span newlines.
    heading_re = re.compile(
        r"(?m)^(?:#{1,4}\s+.+|[A-Z][A-Z\t ]{5,}$|(?:\d+\.\s+[A-Z].+)$)"
    )
    splits = [m.start() for m in heading_re.finditer(doc.text)]
    if not splits:
        return _recursive_semantic(doc, chunk_size, overlap)
    splits.append(len(doc.text))
    chunks = []
    for i in range(len(splits) - 1):
        section = doc.text[splits[i] : splits[i + 1]]
        chunks.extend(
            _section_to_chunks(section, doc, chunk_size, overlap, len(chunks), splits[i])
        )
    return chunks


def _recursive_semantic(doc, chunk_size, overlap):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " "],
        length_function=len,
        keep_separator=True,
    )
    texts = splitter.split_text(doc.text)
    chunks: list[dict] = []
    cursor = 0
    for i, t in enumerate(texts):
        cs = doc.text.find(t, cursor)
        if cs == -1:
            # The splitter can reshape whitespace so the chunk text is not a
            # byte-equal substring. Fall back to a stripped head match.
            head = t.strip()[:30]
            if head:
                cs = doc.text.find(head, cursor)
        if cs == -1:
            # Could not align — skip rather than emit garbage offsets.
            continue
        ce = min(cs + len(t), len(doc.text))
        # Advance cursor past the end of this chunk minus overlap, so we don't
        # re-find the same chunk text on the next iteration.
        cursor = max(cursor, ce - max(overlap, 0))
        chunks.append(_make_chunk(doc.doc_id, i, t, cs, ce))
    return chunks


def _section_to_chunks(section_text, doc, chunk_size, overlap, base_idx, base_char):
    """Convert a section into chunks. `base_char` is the absolute offset of the
    section in `doc.text` (passed in by the caller — never recomputed via find)."""
    nlp = _get_nlp()
    spacy_doc = nlp(section_text)
    # Capture (text, start_char) pairs so we can compute absolute offsets.
    sent_records = [
        (s.text.strip(), int(s.start_char))
        for s in spacy_doc.sents
        if s.text.strip()
    ]
    if not sent_records:
        return []

    chunks: list[dict] = []
    current_sents: list[str] = []
    current_starts: list[int] = []
    current_len = 0

    def flush():
        if not current_sents:
            return
        ct = " ".join(current_sents)
        cs = base_char + current_starts[0]
        ce = cs + len(ct)
        chunks.append(
            _make_chunk(doc.doc_id, base_idx + len(chunks), ct, cs, ce, sentences=list(current_sents))
        )

    for sent_text, sent_local_start in sent_records:
        # Hard-split sentences that exceed the chunk size on their own.
        if len(sent_text) > chunk_size and not current_sents:
            for piece_start in range(0, len(sent_text), chunk_size):
                piece = sent_text[piece_start : piece_start + chunk_size]
                cs = base_char + sent_local_start + piece_start
                ce = cs + len(piece)
                chunks.append(
                    _make_chunk(
                        doc.doc_id, base_idx + len(chunks), piece, cs, ce,
                        sentences=[piece],
                    )
                )
            continue

        # +1 for the joining space we will introduce.
        next_len = current_len + len(sent_text) + (1 if current_sents else 0)
        if next_len > chunk_size and current_sents:
            flush()
            # Build overlap window from the *tail* of current_sents.
            ov_sents: list[str] = []
            ov_starts: list[int] = []
            ov_len = 0
            for s, st in zip(reversed(current_sents), reversed(current_starts)):
                added = len(s) + (1 if ov_sents else 0)
                if ov_len + added > overlap:
                    break
                ov_sents.insert(0, s)
                ov_starts.insert(0, st)
                ov_len += added
            current_sents = ov_sents
            current_starts = ov_starts
            current_len = ov_len

        current_sents.append(sent_text)
        current_starts.append(sent_local_start)
        current_len += len(sent_text) + (1 if len(current_sents) > 1 else 0)

    flush()
    return chunks


def _make_chunk(doc_id, idx, text, cs, ce, sentences=None):
    nlp = _get_nlp()
    spacy_doc = nlp(text)
    if sentences is None:
        sentences = [s.text.strip() for s in spacy_doc.sents if s.text.strip()]
    chunk = {
        # Wider zero-pad to avoid sort-order surprises past 9999.
        "chunk_id": f"{doc_id}_c{idx:06d}",
        "doc_id": doc_id,
        "text": text,
        "sentences": sentences,
        "tokens": [t.text for t in spacy_doc],
        "char_start": cs,
        "char_end": ce,
    }
    return annotate_chunk_with_source(_CURRENT_DOC_BY_ID.get(doc_id), chunk) if doc_id in _CURRENT_DOC_BY_ID else chunk


_CURRENT_DOC_BY_ID: dict[str, Document] = {}


def _with_current_doc(doc: Document, fn, *args):
    _CURRENT_DOC_BY_ID[doc.doc_id] = doc
    try:
        return fn(*args)
    finally:
        _CURRENT_DOC_BY_ID.pop(doc.doc_id, None)


# Backwards compatibility for any module that still references `NLP`.
def __getattr__(name):
    if name == "NLP":
        return _get_nlp()
    raise AttributeError(name)
