"""Text normalization utilities."""

from __future__ import annotations

import re
import unicodedata

# Bullet glyphs to map to "- ". Constrained to NON-newline whitespace afterwards
# so a bullet at end-of-line does not eat the newline boundary.
_BULLET_RE = re.compile(r"[•·▪▸►➢●‣◦▫❑■][ \t]*")

# PDF line-wrap dehyphenation: "re-\ntrieval" -> "retrieval".
# Applied before newline collapse so we don't accidentally re-introduce the gap.
_HYPHEN_LINEWRAP_RE = re.compile(r"(\w)-\n(\w)")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xad", "").replace("\x0c", "\n")
    text = _HYPHEN_LINEWRAP_RE.sub(r"\1\2", text)
    text = _BULLET_RE.sub("- ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
