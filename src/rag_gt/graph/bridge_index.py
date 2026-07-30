"""Stage A — Bridge index.

Give every fact a set of *bridge keys* (salient entities / multiword terms /
standard references) and find facts that SHARE a key across DIFFERENT pages.
This is the deterministic ($0) foundation of the multi-hop instrument: a bridge
is a real shared entity verifiable in both facts, not an LLM-invented relation.

Design choice (our fix to GRADE's surface-string entity merging): bridge keys are
salient phrases drawn from the facts themselves — standard refs by regex, and
multiword noun-ish phrases by content n-grams — normalized so plural/spacing/hyphen
variants collapse. No generic NER model required; runs offline at zero cost.

CLI:  PYTHONPATH=src python -m rag_gt.graph.bridge_index <s4_facts.json> [out.json]
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from typing import Dict, List

# Standard references (ISO/EN/DIN/IEC/ASTM/BS/AWS, incl. /TR /TS), e.g. "ISO 15607",
# "ISO/TR 18491", "DIN EN ISO 15609-1".
_STD_REF_RE = re.compile(
    r"\b(?:DIN\s+)?(?:EN\s+)?(?:ISO|IEC|ASTM|BS|AWS)(?:/TR|/TS)?\s*\d{3,5}(?:[-‐‑]\d+)?",
    re.I,
)

# Generic tokens that never anchor a bridge on their own.
_STOP = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "is", "are", "be",
    "shall", "with", "by", "as", "at", "from", "which", "that", "this", "these",
    "those", "such", "where", "when", "if", "any", "all", "also", "each", "include",
    "includes", "including", "given", "made", "make", "used", "use", "provide",
    "provides", "specified", "specify", "information", "document", "requirement",
    "requirements", "applicable", "appropriate", "according", "relevant", "other",
    "e.g", "eg", "i.e", "ie", "etc", "details", "detail",
}


def _norm_token(t: str) -> str:
    """Lowercase + strip a single trailing plural 's' (specifications -> specification)."""
    t = t.lower()
    if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    return t


def _tokens(text: str) -> List[str]:
    # split on anything that is not a letter/digit; '/' and hyphens become breaks
    return [w for w in re.split(r"[^A-Za-z0-9]+", text) if w]


# Citation / running-header phrase fragments that are never bridge entities.
# (Header/footer watermarks should be stripped upstream in Stage 4; this is a
# defensive net for stale fact snapshots — page header text is never an entity.)
_BOILERPLATE = ("accordance", "printed copies", "copies are uncontrolled")


def _is_salient_phrase(toks: List[str]) -> bool:
    if _norm_token(toks[0]) in _STOP or _norm_token(toks[-1]) in _STOP:
        return False
    if all(_norm_token(t) in _STOP for t in toks):
        return False
    # must contain >=1 substantive ALPHABETIC token (rejects pure-number fragments
    # like "15609 1 2019 12" — those are covered by the STANDARD_REF key instead)
    return any(t.isalpha() and len(t) >= 4 and _norm_token(t) not in _STOP for t in toks)


def extract_bridge_keys(text: str) -> List[Dict[str, str]]:
    """Return [{type, surface, norm}] bridge keys for one fact's text."""
    keys: Dict[str, Dict[str, str]] = {}

    # 1) standard references
    for m in _STD_REF_RE.finditer(text):
        surface = re.sub(r"\s+", " ", m.group(0)).strip()
        norm = re.sub(r"[‐‑]", "-", surface.lower())
        norm = re.sub(r"\s+", " ", norm)
        keys.setdefault(("STANDARD_REF", norm), {"type": "STANDARD_REF", "surface": surface, "norm": norm})

    # 2) multiword phrase keys (2- to 4-grams of content tokens)
    toks = _tokens(text)
    for n in (4, 3, 2):
        for i in range(len(toks) - n + 1):
            window = toks[i:i + n]
            if not _is_salient_phrase(window):
                continue
            norm = " ".join(_norm_token(t) for t in window)
            if any(b in norm for b in _BOILERPLATE):
                continue
            surface = " ".join(window)
            keys.setdefault(("TERM", norm), {"type": "TERM", "surface": surface, "norm": norm})

    return list(keys.values())


def _get(f: dict, *names, default=None):
    for n in names:
        if n in f and f[n] is not None:
            return f[n]
    return default


def _doc_of(f: dict) -> str:
    """Document id: explicit `doc` field, else the fact_id prefix, else 'doc'."""
    d = _get(f, "doc", "doc_id")
    if d:
        return d
    fid = _get(f, "fact_id", "id", default="")
    return re.sub(r"_F?\d+.*$", "", fid) or "doc"


# A genuine bridge connects a FEW facts on a FEW pages. A phrase that recurs on
# many pages is a running header/watermark ("Printed copies are uncontrolled") or a
# document-wide theme ("reinforcement learning") — never a specific bridge entity.
# This document-frequency band is the one general fix that suppresses both, across
# every doc type, without hardcoding any watermark or topic string.
_MAX_BRIDGE_PAGES = 4


def build_bridge_index(facts: List[dict], max_pages: int = _MAX_BRIDGE_PAGES) -> dict:
    """Build the bridge index. Schema-agnostic (handles both the s4_facts shape
    `{fact_id, canonical_form, page_start}` and the all_facts shape
    `{id, text, doc, page}`). Bridges are formed WITHIN a document only — a shared
    entity on two pages of the SAME doc. (Cross-document bridges are a later stage.)
    A bridge must span 2..``max_pages`` pages; more = header/theme, dropped.
    """
    fact_bridges: Dict[str, List[dict]] = {}
    # key inversion is per-document: (doc, norm) -> set(fact_id)
    norm_to_facts: Dict[tuple, set] = defaultdict(set)
    norm_meta: Dict[tuple, dict] = {}
    fact_page: Dict[str, int] = {}

    for f in facts:
        fid = _get(f, "fact_id", "id")
        text = _get(f, "canonical_form", "text", default="")
        page = _get(f, "page_start", "page")
        doc = _doc_of(f)
        fact_page[fid] = page
        ks = extract_bridge_keys(text)
        fact_bridges[fid] = ks
        for k in ks:
            gk = (doc, k["norm"])
            norm_to_facts[gk].add(fid)
            norm_meta.setdefault(gk, k)

    # cross-page groups only: a key whose facts span >= 2 distinct pages in one doc
    groups = []
    for (doc, norm), fids in norm_to_facts.items():
        pages = sorted({fact_page[fid] for fid in fids if fact_page[fid] is not None})
        if len(fids) >= 2 and 2 <= len(pages) <= max_pages:
            groups.append({
                "doc": doc,
                "norm": norm,
                "type": norm_meta[(doc, norm)]["type"],
                "surface": norm_meta[(doc, norm)]["surface"],
                "fact_ids": sorted(fids),
                "pages": pages,
            })
    # Collapse redundant sub-phrases: many n-gram keys map to the SAME fact-set
    # (e.g. "welding procedure" / "procedure specifications" / "welding procedure
    # specifications"). Keep ONE representative per fact-set — a STANDARD_REF if
    # present, otherwise the longest surface phrase (most distinctive entity).
    by_set: Dict[frozenset, dict] = {}
    for g in groups:
        key = frozenset(g["fact_ids"])
        cur = by_set.get(key)
        if cur is None:
            by_set[key] = g
            continue
        better = (
            (g["type"] == "STANDARD_REF") > (cur["type"] == "STANDARD_REF")
            or (g["type"] == cur["type"] and len(g["surface"]) > len(cur["surface"]))
        )
        if better:
            by_set[key] = g
    collapsed = list(by_set.values())
    # most-distinctive first: standard refs and longer phrases on top
    collapsed.sort(key=lambda g: (g["type"] != "STANDARD_REF", -len(g["surface"])))

    return {
        "fact_bridges": fact_bridges,
        "bridge_groups": collapsed,
        "stats": {
            "n_facts": len(facts),
            "n_keys": len(norm_to_facts),
            "n_cross_page_groups": len(collapsed),
        },
    }


def _main(argv: List[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    facts = json.loads(open(argv[0], encoding="utf-8").read())
    if isinstance(facts, dict) and "facts" in facts:
        facts = facts["facts"]
    idx = build_bridge_index(facts)
    out = argv[1] if len(argv) > 1 else None

    s = idx["stats"]
    print(f"facts={s['n_facts']}  keys={s['n_keys']}  cross_page_groups={s['n_cross_page_groups']}\n")
    by_doc: Dict[str, list] = defaultdict(list)
    for g in idx["bridge_groups"]:
        by_doc[g.get("doc", "doc")].append(g)
    print("=== CROSS-PAGE BRIDGE GROUPS per document ===")
    for doc in sorted(by_doc, key=lambda d: -len(by_doc[d])):
        gs = by_doc[doc]
        print(f"\n## {doc}  ({len(gs)} bridges)")
        for g in gs[:12]:
            short = [fid.split("_")[-1] for fid in g["fact_ids"]]
            print(f"  [{g['type']}] '{g['surface']}'  pages={g['pages']}  facts={short}")
        if len(gs) > 12:
            print(f"  ... +{len(gs)-12} more")

    if out:
        json.dump(idx, open(out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
