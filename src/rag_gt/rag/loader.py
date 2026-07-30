"""CP1 — GT pairs + chunk loader with optional re-chunking strategies.

Responsibilities:
  - load_gt_pairs(doc_id)      → list of GT pair dicts for one document
  - load_chunks(doc_id)        → list of chunk dicts from s2_chunks_full.json
  - rechunk(chunks, strategy)  → list of re-chunked dicts for alternate strategies
  - load_all_gt_pairs()        → all 657 pairs, grouped by doc_id

Chunking strategies
  "original"    - return s2_chunks_full.json as-is (baseline)
  "sentence"    - split each chunk by sentences (spaCy sentencizer)
  "sliding_256" - 256-token sliding window, 128-token overlap
  "paragraph"   - split by double newline, min 50 chars
"""
from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]   # repo root
# Generated chunk data is large and untracked; a git worktree does not carry
# it. RAG_GT_DATA_ROOT points chunk loading at the checkout that has it
# (GT pairs stay on _REPO_ROOT — they are version-controlled per branch).
_DATA_ROOT = Path(os.environ.get("RAG_GT_DATA_ROOT", str(_REPO_ROOT)))
_PIPELINE_BASE = _DATA_ROOT / "data" / "test_corpus_allpdf" / "pipeline_run"
_GT_FILE = _REPO_ROOT / "data" / "eval_results" / "allpdf_phase2_pairs.jsonl"

# doc_id  →  phase2 directory name
_DOC_DIR_MAP: Dict[str, str] = {
    "attention_transformer":           "attention_transformer_phase2",
    "berkshire_2022ar":                "berkshire_2022ar_phase2",
    "bert":                            "bert_phase2",
    "bonaventure_net":                 "bonaventure_net_phase2",
    "din_iso_15609_welding_procedure": "din_iso_15609_welding_procedure_phase2",
    "din_iso_3834_welding_quality":    "din_iso_3834_welding_quality_phase2",
    "din_iso_6507_vickers":            "din_iso_6507_vickers_phase2",
    "ecma404_json":                    "ecma404_json_phase2",
    "pmc_biomed":                      "pmc_biomed_phase2",
    "rag_lewis2020":                   "rag_lewis2020_phase2",
    "requirements_eng":                "requirements_eng_phase2",
    "sutton_rl":                       "sutton_rl_phase2",
    "synth_scanned":                   "synth_scanned_phase2",
}

ALL_DOC_IDS: List[str] = sorted(_DOC_DIR_MAP.keys())

# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_gt_pairs(doc_id: str) -> List[dict]:
    """Return GT pairs whose facts belong to doc_id."""
    pairs: List[dict] = []
    text = _GT_FILE.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        pair = json.loads(line)
        # Infer doc_id from the first fact_id by stripping the trailing _FXXXXXX
        first_fid = (pair.get("chain_fact_ids") or [""])[0]
        # fact_id format: {doc_id}_F{number}
        inferred = _doc_id_from_fact_id(first_fid)
        if inferred == doc_id:
            pairs.append(pair)
    return pairs


def load_all_gt_pairs() -> Dict[str, List[dict]]:
    """Return {doc_id: [pairs]} for all 13 documents."""
    grouped: Dict[str, List[dict]] = {d: [] for d in ALL_DOC_IDS}
    text = _GT_FILE.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        pair = json.loads(line)
        first_fid = (pair.get("chain_fact_ids") or [""])[0]
        doc_id = _doc_id_from_fact_id(first_fid)
        if doc_id in grouped:
            grouped[doc_id].append(pair)
    return grouped


def load_chunks(doc_id: str) -> List[dict]:
    """Load s2_chunks_full.json for doc_id."""
    dir_name = _DOC_DIR_MAP.get(doc_id)
    if dir_name is None:
        raise ValueError(f"Unknown doc_id: {doc_id!r}. Known: {list(_DOC_DIR_MAP)}")
    path = _PIPELINE_BASE / dir_name / "checkpoints" / "s2_chunks_full.json"
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def rechunk(chunks: List[dict], strategy: str) -> List[dict]:
    """Apply an alternative chunking strategy to the original chunks.

    All strategies return dicts with at minimum:
      chunk_id, doc_id, text, page_start, page_end
    """
    if strategy == "original":
        return chunks
    if strategy == "sentence":
        return _rechunk_sentence(chunks)
    if strategy == "sliding_256":
        return _rechunk_sliding(chunks, window=256, overlap=128)
    if strategy == "paragraph":
        return _rechunk_paragraph(chunks)
    raise ValueError(f"Unknown chunking strategy: {strategy!r}. "
                     "Choose from: original, sentence, sliding_256, paragraph")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FACT_ID_RE = re.compile(r"^(.+?)_F\d+$")

def _doc_id_from_fact_id(fact_id: str) -> str:
    """Extract doc_id from fact_id like 'din_iso_15609_welding_procedure_F021001'."""
    m = _FACT_ID_RE.match(fact_id)
    return m.group(1) if m else fact_id


def _make_chunk(doc_id: str, strategy: str, idx: int, text: str,
                page_start: Optional[int], page_end: Optional[int]) -> dict:
    return {
        "chunk_id": f"{doc_id}_rechunk_{strategy}_{idx:06d}",
        "doc_id": doc_id,
        "text": text,
        "page_start": page_start,
        "page_end": page_end,
    }


# --- Sentence strategy ---

def _rechunk_sentence(chunks: List[dict]) -> List[dict]:
    """Split each chunk into sentences using a simple regex sentencizer.

    Falls back to spaCy sentencizer if available.  If a chunk produces no
    sentences (e.g. table-of-contents line), the whole chunk is kept as-is.
    """
    try:
        return _rechunk_sentence_spacy(chunks)
    except Exception:
        return _rechunk_sentence_regex(chunks)


def _rechunk_sentence_spacy(chunks: List[dict]) -> List[dict]:
    import spacy  # noqa: F401
    from functools import lru_cache

    @lru_cache(maxsize=1)
    def _get_nlp():
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])
        except OSError:
            nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp

    nlp = _get_nlp()
    out: List[dict] = []
    doc_id = chunks[0]["doc_id"] if chunks else ""
    idx = 0
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        page_s = chunk.get("page_start")
        page_e = chunk.get("page_end")
        try:
            doc = nlp(text)
            sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) >= 30]
        except Exception:
            sents = []
        if not sents:
            out.append(_make_chunk(doc_id, "sentence", idx, text, page_s, page_e))
            idx += 1
            continue
        for sent in sents:
            out.append(_make_chunk(doc_id, "sentence", idx, sent, page_s, page_e))
            idx += 1
    return out


_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _rechunk_sentence_regex(chunks: List[dict]) -> List[dict]:
    out: List[dict] = []
    doc_id = chunks[0]["doc_id"] if chunks else ""
    idx = 0
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        page_s = chunk.get("page_start")
        page_e = chunk.get("page_end")
        sents = [s.strip() for s in _SENT_RE.split(text) if len(s.strip()) >= 30]
        if not sents:
            out.append(_make_chunk(doc_id, "sentence", idx, text, page_s, page_e))
            idx += 1
            continue
        for sent in sents:
            out.append(_make_chunk(doc_id, "sentence", idx, sent, page_s, page_e))
            idx += 1
    return out


# --- Sliding window strategy ---

_TOKEN_RE_SLIDING = re.compile(r"\S+")


def _rechunk_sliding(chunks: List[dict], window: int = 256, overlap: int = 128) -> List[dict]:
    """Concatenate all chunk texts, split into token-window chunks."""
    if not chunks:
        return []
    doc_id = chunks[0]["doc_id"]

    # Build flat token list, tracking which original chunk each token belongs to.
    all_tokens: List[str] = []
    token_page: List[Optional[int]] = []   # page_start for each token

    for chunk in chunks:
        text = chunk.get("text", "")
        page = chunk.get("page_start")
        toks = _TOKEN_RE_SLIDING.findall(text)
        all_tokens.extend(toks)
        token_page.extend([page] * len(toks))

    if not all_tokens:
        return []

    out: List[dict] = []
    step = window - overlap
    if step <= 0:
        step = 1
    idx = 0
    i = 0
    while i < len(all_tokens):
        chunk_toks = all_tokens[i : i + window]
        page_s = token_page[i]
        page_e = token_page[min(i + window - 1, len(token_page) - 1)]
        text = " ".join(chunk_toks)
        if text.strip():
            out.append(_make_chunk(doc_id, "sliding_256", idx, text, page_s, page_e))
            idx += 1
        i += step

    return out


# --- Paragraph strategy ---

def _rechunk_paragraph(chunks: List[dict]) -> List[dict]:
    """Split each chunk by double newline paragraphs, min 50 chars."""
    out: List[dict] = []
    doc_id = chunks[0]["doc_id"] if chunks else ""
    idx = 0
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        page_s = chunk.get("page_start")
        page_e = chunk.get("page_end")
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= 50]
        if not paras:
            out.append(_make_chunk(doc_id, "paragraph", idx, text, page_s, page_e))
            idx += 1
            continue
        for para in paras:
            out.append(_make_chunk(doc_id, "paragraph", idx, para, page_s, page_e))
            idx += 1
    return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== CP1 Loader Smoke Test ===")

    # Test GT loading
    for test_doc in ["ecma404_json", "din_iso_15609_welding_procedure"]:
        pairs = load_gt_pairs(test_doc)
        chunks = load_chunks(test_doc)
        print(f"\n[{test_doc}]")
        print(f"  GT pairs: {len(pairs)}")
        print(f"  Original chunks: {len(chunks)}")

        for strategy in ["original", "sentence", "sliding_256", "paragraph"]:
            rechunked = rechunk(chunks, strategy)
            print(f"  {strategy}: {len(rechunked)} chunks")

    # Verify all doc_ids can be loaded
    print("\n[all-docs GT pairs]")
    all_pairs = load_all_gt_pairs()
    total = 0
    for doc_id, ps in sorted(all_pairs.items()):
        total += len(ps)
        print(f"  {doc_id}: {len(ps)} pairs")
    print(f"  TOTAL: {total} pairs")
