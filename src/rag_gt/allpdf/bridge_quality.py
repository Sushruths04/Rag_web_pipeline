"""Bridge surface quality gate (Phase 1 debug plan T2, kills finding F1).

The round-2 shipped dataset carried 10 multi-hop pairs behind 9 distinct junk
bridge surfaces -- OCR fragments ("di erent", "nite number"), comparative
fillers ("less than", "than 0"), and vague quantifiers ("some number", "n
step"). None of these name a real, checkable concept; a bridge surface is
supposed to be the thing the two pages actually share.

``surface_ok`` is a free (no-LLM), deterministic gate applied to a bridge
pair's ``bridge_entity``/``bridge_norm`` BEFORE it is allowed to seed a
cluster or fall back to the v1 2-fact path. Two known OCR-split patterns are
REPAIRED and re-tested rather than blind-rejected (``di erent`` -> different,
a ``nite`` prefix fragment -> finite); every other junk surface is rejected
outright via an explicit seed list plus two structural checks (too short, no
real content token).
"""
from __future__ import annotations

import re
import unicodedata

# The 9 junk bridge surfaces measured in the round-2 dataset (F1), normalized
# (casefold + collapsed whitespace) for matching. Two of these have a known
# OCR-split repair (see _OCR_REPAIRS / _OCR_PREFIX_REPAIRS below) and are
# retested after repair instead of being rejected on sight.
JUNK_SURFACES: frozenset[str] = frozenset(
    {
        "backed up",
        "di erent",
        "less than",
        "less than 5",
        "n step",
        "nite number",
        "only in the limit",
        "some number",
        "than 0",
    }
)

# Whole-surface OCR-split fixes: the surface, once normalized, is replaced
# outright.
_OCR_REPAIRS: dict[str, str] = {
    "di erent": "different",
}

# Per-token prefix-fragment fixes: a token that IS exactly the fragment gets
# replaced (e.g. "nite" -> "finite" inside "nite number").
_OCR_PREFIX_REPAIRS: dict[str, str] = {
    "nite": "finite",
}

_MIN_LEN = 5
_MIN_CONTENT_TOKEN_LEN = 3


def _normalize(surface: str) -> str:
    text = unicodedata.normalize("NFKC", surface)
    return " ".join(text.split())


def _has_content_token(surface: str) -> bool:
    # A maximal alphabetic run of >=3 chars, e.g. "ISO" in "ISO 15607" or
    # "Vickers" in "Vickers microhardness" -- long enough to be a real word
    # or acronym, short enough not to exclude standard identifiers.
    return any(
        len(tok) >= _MIN_CONTENT_TOKEN_LEN for tok in re.findall(r"[A-Za-z]+", surface)
    )


def _structural_ok(surface: str) -> tuple[bool, str]:
    if len(surface) < _MIN_LEN:
        return False, "too_short"
    if not _has_content_token(surface):
        return False, "no_content_token"
    return True, "ok"


def _try_repair(normalized: str) -> str | None:
    """Return a repaired surface if a known OCR-split pattern matches, else
    None. Does not itself decide accept/reject -- the caller re-tests the
    repaired text through the normal rules."""
    lower = normalized.lower()
    whole = _OCR_REPAIRS.get(lower)
    if whole is not None:
        return whole
    tokens = normalized.split(" ")
    changed = False
    fixed_tokens = []
    for tok in tokens:
        repl = _OCR_PREFIX_REPAIRS.get(tok.lower())
        if repl is not None:
            fixed_tokens.append(repl)
            changed = True
        else:
            fixed_tokens.append(tok)
    if changed:
        return " ".join(fixed_tokens)
    return None


def surface_ok(surface: object) -> tuple[bool, str]:
    """Return (ok, detail).

    ``ok is True``: ``detail`` is the surface to use (repaired if a known
    OCR-split pattern applied, otherwise the normalized input unchanged).
    ``ok is False``: ``detail`` is a short machine-readable rejection reason
    (``"empty"``, ``"junk_surface"``, ``"too_short"``, ``"no_content_token"``).
    """
    normalized = _normalize(str(surface or ""))
    if not normalized:
        return False, "empty"

    repaired = _try_repair(normalized)
    if repaired is not None:
        ok, reason = _structural_ok(repaired)
        if ok and repaired.strip().lower() not in JUNK_SURFACES:
            return True, repaired
        # repair did not produce a usable surface -- fall through to the
        # normal reject path on the ORIGINAL text.

    if normalized.lower() in JUNK_SURFACES:
        return False, "junk_surface"

    ok, reason = _structural_ok(normalized)
    return (True, normalized) if ok else (False, reason)
