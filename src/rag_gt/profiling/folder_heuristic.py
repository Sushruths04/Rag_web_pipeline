"""Layer-0 path-based DocType fast-path. Runs before the regex profiler."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from rag_gt.core.types import DocType

FOLDER_MAP: dict[str, DocType] = {
    "iso": "ISO_STANDARD",
    "standards": "ISO_STANDARD",
    "din": "ISO_STANDARD",
    "iec": "ISO_STANDARD",
    "specs": "ISO_STANDARD",
    "specifications": "ISO_STANDARD",
    "reports": "REPORT",
    "annual": "REPORT",
    "financials": "REPORT",
    "quarterly": "REPORT",
    "internal": "KT_DOC",
    "kt": "KT_DOC",
    "handover": "KT_DOC",
    "onboarding": "KT_DOC",
    "knowledge": "KT_DOC",
    "narrative": "NARRATIVE",
    "fiction": "NARRATIVE",
    "stories": "NARRATIVE",
}

FILENAME_PREFIXES: list[tuple[re.Pattern, DocType]] = [
    (re.compile(r"^DIN[_ -]?EN[_ -]?ISO", re.I), "ISO_STANDARD"),
    (re.compile(r"^ISO[_ -]?\d", re.I), "ISO_STANDARD"),
    (re.compile(r"^IEC[_ -]?\d", re.I), "ISO_STANDARD"),
    (re.compile(r"^IEEE[_ -]?\d", re.I), "ISO_STANDARD"),
    (re.compile(r"^RFC[_ -]?\d", re.I), "ISO_STANDARD"),
    (re.compile(r"^RPT[_ -]", re.I), "REPORT"),
    (re.compile(r"^ANN[_ -]?RPT", re.I), "REPORT"),
    (re.compile(r"^KT[_ -]", re.I), "KT_DOC"),
    (re.compile(r"^HANDOVER[_ -]", re.I), "KT_DOC"),
]


def infer_from_path(path: Path | str) -> Optional[DocType]:
    p = Path(path)
    name = p.name
    for pattern, doc_type in FILENAME_PREFIXES:
        if pattern.search(name):
            return doc_type
    for part in p.parts:
        key = part.lower()
        if key in FOLDER_MAP:
            return FOLDER_MAP[key]
    return None
