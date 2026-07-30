"""Retroactively apply the hardened Stage-4 cleaner to an existing fact dump.

The full re-extraction through Stage 0-4 is the gold path but is hours-slow on a
large corpus. This pass gets the SAME watermark/junk removal at $0 by running the
*same* hardened detectors over facts that were extracted before those fixes landed
(e.g. the stale ``all_facts.json``). It does NOT re-decontextualize — it strips
running-header watermarks from the text and drops structural-junk facts.

CLI:  PYTHONPATH=src python -m rag_gt.allpdf.clean_facts <in_facts.json> <out_facts.json>
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from typing import List

from rag_gt.facts.domain_filter import (
    has_dangling_anaphora,
    is_bare_numbered_heading,
    is_printout_metadata,
    is_reference_list_dump,
    is_table_artifact,
    strip_running_artifacts,
)

# extra watermark stems seen in the ISO controlled-copy snapshots
_WATERMARK_STEMS = ("printed copies are uncontrolled", "user name: ip")


def _reject(text: str) -> str | None:
    t = " ".join((text or "").split())
    if len(t) < 25:
        return "too_short"
    low = t.lower()
    if any(s in low for s in _WATERMARK_STEMS):
        return "watermark"
    if is_printout_metadata(t):
        return "printout_metadata"
    if is_table_artifact(t):
        return "table_artifact"
    if is_reference_list_dump(t):
        return "reference_list_dump"
    if is_bare_numbered_heading(t):
        return "bare_numbered_heading"
    if has_dangling_anaphora(t):
        return "dangling_anaphora"
    return None


def clean_facts(facts: List[dict]) -> tuple[List[dict], dict]:
    kept: List[dict] = []
    dropped: Counter = Counter()
    for f in facts:
        raw = f.get("canonical_form") or f.get("text") or ""
        cleaned = strip_running_artifacts(raw)
        reason = _reject(cleaned)
        if reason:
            dropped[reason] += 1
            continue
        g = dict(f)
        # write the cleaned text back into both fields used downstream
        if "text" in g:
            g["text"] = cleaned
        if "canonical_form" in g:
            g["canonical_form"] = cleaned
        kept.append(g)
    return kept, dict(dropped)


def _main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    facts = json.loads(open(argv[0], encoding="utf-8").read())
    if isinstance(facts, dict) and "facts" in facts:
        facts = facts["facts"]
    kept, dropped = clean_facts(facts)
    print(f"in={len(facts)}  kept={len(kept)}  dropped={sum(dropped.values())}  reasons={dropped}")
    json.dump(kept, open(argv[1], "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"wrote {argv[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
