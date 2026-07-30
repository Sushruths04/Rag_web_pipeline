"""Structural quality scoring for facts (zero domain keywords).

The score returned is in [0, 1]. PDF-extraction artifacts (Unicode replacement
characters, CID glyph references, reversed page-header text, low letter
density) score 0.0 — they cannot be facts and must not enter the GT.
"""

from __future__ import annotations

import re

# PDF artifact patterns.
_REPLACEMENT_CHAR_RE = re.compile(r"�")
_CID_RE = re.compile(r"\bcid:\s*\d+\b", re.I)

# Pages that get scanned upside down or that contain rotated watermarks
# produce reversed text that lexically reverses to common English. We don't
# need a full reversed-corpus check; flagging a small set of high-frequency
# tokens is enough to catch the dominant failure mode in practice.
_REVERSED_TOKENS = {
    "dellortnocnu",  # uncontrolled
    "seipoc",        # copies
    "detnirP",       # Printed
    "egap",          # page
    "noisrev",       # version
    "tnemucod",      # document
    "krameremos",    # somewatermark (sentinel)
}


def _looks_like_pdf_artifact(text: str) -> bool:
    """Return True if the text exhibits a known PDF-extraction failure mode."""
    if not text:
        return True
    if _REPLACEMENT_CHAR_RE.search(text):
        return True
    if _CID_RE.search(text):
        return True
    lowered = text.lower()
    tokens = re.findall(r"\b\w+\b", lowered)
    if any(tok in _REVERSED_TOKENS for tok in tokens):
        return True
    if len(tokens) < 3:
        return True
    letters = sum(1 for c in text if c.isalpha())
    if letters / max(len(text), 1) < 0.6:
        return True
    distinct = len(set(tokens))
    if distinct < 3:
        return True
    return False


def structural_quality(text: str) -> float:
    """Score a candidate fact in [0, 1]. PDF artifacts return 0.0."""
    if _looks_like_pdf_artifact(text):
        return 0.0
    score = 0.5
    if any(c in text for c in [".", ":", ";"]):
        score += 0.1
    if re.search(
        r"\b(is|are|was|were|shall|must|should|can|could|will|would)\b", text, re.I
    ):
        score += 0.1
    if re.search(r"\b(not|no|never|neither|nor)\b", text, re.I):
        score += 0.1
    if len(text.split()) >= 5:
        score += 0.1
    if len(text.split()) >= 10:
        score += 0.05
    if re.search(r"\d+", text):
        score += 0.05
    return min(score, 1.0)
