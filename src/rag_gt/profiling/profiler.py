"""Rule-based document profiler. Layer-0 path heuristic runs first (optional)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from rag_gt.core.config import load_config
from rag_gt.core.types import Document
from rag_gt.profiling.folder_heuristic import infer_from_path

_SIGNALS: dict[str, list[str]] = {
    "ISO_STANDARD": [
        r"\b(shall|must|should not|is required to)\b",
        r"\b(iso|iec|ieee|rfc|specification|standard|annex|clause)\b",
        r"\b(tolerance|threshold|compliance|parameter|requirement)\b",
        r"§\s*\d",
        r"\d+\.\d+\.\d+",
    ],
    "REPORT": [
        r"\b(quarterly|annual report|fiscal year|ebitda|revenue|shareholders)\b",
        r"\b(board of directors|ceo|cfo|vp |headcount|workforce)\b",
        r"\b(policy|procedure|audit|regulation)\b",
        r"\$\s*\d+[.,]?\d*\s*(million|billion|m\b|bn\b)",
    ],
    "NARRATIVE": [
        r"\b(said|told|reported|according to|argued|claimed)\b",
        r"\b(story|chapter|narrative|protagonist)\b",
    ],
    "KT_DOC": [
        r"\b(lesson|learned|knowledge|transfer|onboarding|handover)\b",
        r"\b(team|project|process|workflow|escalation)\b",
    ],
    "TEXTBOOK": [
        r"\b(theorem|lemma|corollary|proof|definition|example)\b",
        r"\b(algorithm|pseudocode|gradient|derivative)\b",
        r"\b(reward|policy|state|action|agent|MDP|RL)\b",
        r"\b(equation \d+|eq\.?\s*\d+|figure \d+|table \d+)\b",
    ],
}


def profile_document(
    doc: Document,
    path: Optional[Path | str] = None,
    enable_fast_path: Optional[bool] = None,
) -> dict:
    if enable_fast_path is None:
        cfg = load_config()
        enable_fast_path = cfg.get("profiling", {}).get("enable_folder_heuristic", True)

    fast_path_hit: Optional[str] = None
    if enable_fast_path and path is not None:
        fast_path_hit = infer_from_path(path)

    sample = doc.text[:2000].lower()

    def count(patterns):
        return sum(len(re.findall(p, sample)) for p in patterns)

    scores = {dt: count(pats) for dt, pats in _SIGNALS.items()}
    winner = max(scores, key=scores.get)

    if fast_path_hit is not None:
        refined = fast_path_hit
    elif scores[winner] >= 3:
        refined = winner
    elif doc.doc_type != "UNKNOWN":
        refined = doc.doc_type  # CLI hint is last resort
    else:
        refined = "UNKNOWN"

    return {
        "doc_id": doc.doc_id,
        "doc_type": refined,
        "avg_line_len": len(doc.text) / max(doc.text.count("\n") + 1, 1),
        "signal_scores": scores,
        "char_count": len(doc.text),
        "fast_path_hit": fast_path_hit,
    }
