"""Stage B — Bridge linker.

Turn Stage-A bridge groups into verified multi-hop *pairs*. Deterministic ($0):

  1. enumerate cross-page fact pairs within each bridge group;
  2. ANTI-FABRICATION gate — the bridge entity must appear in BOTH facts
     (guaranteed by construction; re-checked for safety);
  3. DUPLICATE gate — drop pairs whose two facts say the SAME thing (high content
     overlap): a multi-hop needs each fact to contribute something DIFFERENT;
  4. dedup fact-pairs that share several bridges → keep the most distinctive bridge.

The LLM only ever VERIFIES later (Stage C/D); it never invents the link here.

CLI:
  PYTHONPATH=src python -m rag_gt.graph.bridge_linker <all_facts.json> <bridge_index.json> [out.json]
"""
from __future__ import annotations

import json
import re
import sys
from itertools import combinations
from typing import Dict, List

from rag_gt.graph.bridge_index import _get, _norm_token, _tokens, _STOP

# Two facts whose content tokens overlap at/above this fraction are near-duplicates
# (the same statement on two pages) — not a multi-hop.
_DUP_JACCARD = 0.6


def _content_tokens(text: str, drop: set) -> set:
    """Substantive tokens (len>=4, not stopword), minus the bridge's own tokens."""
    out = set()
    for t in _tokens(text):
        n = _norm_token(t)
        if t.isalpha() and len(t) >= 4 and n not in _STOP and n not in drop:
            out.add(n)
    return out


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_bridge_pairs(facts: List[dict], bridge_index: dict) -> dict:
    by_id: Dict[str, dict] = {}
    for f in facts:
        by_id[_get(f, "fact_id", "id")] = f

    # best bridge per unordered fact-pair (most distinctive: STANDARD_REF, longer)
    best: Dict[tuple, dict] = {}
    n_raw = n_drop_dup = n_drop_missing = 0

    for g in bridge_index["bridge_groups"]:
        bnorm = g["norm"]
        bdrop = set(bnorm.split())
        fids = [fid for fid in g["fact_ids"] if fid in by_id]
        for fa, fb in combinations(fids, 2):
            A, B = by_id[fa], by_id[fb]
            pa = _get(A, "page_start", "page")
            pb = _get(B, "page_start", "page")
            if pa == pb:
                continue  # same-page pairs are not multi-page
            ta = _get(A, "canonical_form", "text", default="")
            tb = _get(B, "canonical_form", "text", default="")
            n_raw += 1
            # (2) anti-fabrication: bridge present in BOTH facts.
            # BUG-E: pad with spaces so the check is on whole tokens. A raw
            # substring test lets a spurious cross-token match through, e.g.
            # bnorm "test method" matching inside "greatest method".
            if not _contains_phrase(ta, bnorm) or not _contains_phrase(tb, bnorm):
                n_drop_missing += 1
                continue
            # (3) duplicate gate
            ca = _content_tokens(ta, bdrop)
            cb = _content_tokens(tb, bdrop)
            if _jaccard(ca, cb) >= _DUP_JACCARD:
                n_drop_dup += 1
                continue
            key = tuple(sorted((fa, fb)))
            cand = {
                "doc": g["doc"], "fact_a": key[0], "fact_b": key[1],
                "bridge_entity": g["surface"], "bridge_norm": bnorm,
                "bridge_type": g["type"], "pages": sorted({pa, pb}),
                "content_jaccard": round(_jaccard(ca, cb), 3),
            }
            cur = best.get(key)
            if cur is None or _more_distinctive(cand, cur):
                best[key] = cand

    pairs = list(best.values())
    pairs.sort(key=lambda p: (p["bridge_type"] != "STANDARD_REF", p["content_jaccard"]))
    for i, p in enumerate(pairs, 1):
        p["pair_id"] = f"BP{i:04d}"

    return {
        "pairs": pairs,
        "stats": {
            "candidate_pairs": n_raw,
            "dropped_bridge_missing": n_drop_missing,
            "dropped_duplicate": n_drop_dup,
            "verified_pairs": len(pairs),
        },
    }


def _norm_text(text: str) -> str:
    toks = [_norm_token(t) for t in _tokens(text)]
    return " ".join(toks)


def _contains_phrase(text: str, phrase_norm: str) -> bool:
    """True when `phrase_norm` occurs in `text` on whole-token boundaries.

    Pads both the normalized text and the phrase with spaces so a multi-token
    bridge like "test method" cannot spuriously match inside "greatest method"
    (BUG-E). `phrase_norm` is already normalized by the caller.
    """
    if not phrase_norm:
        return False
    return f" {phrase_norm} " in f" {_norm_text(text)} "


def _more_distinctive(a: dict, b: dict) -> bool:
    if (a["bridge_type"] == "STANDARD_REF") != (b["bridge_type"] == "STANDARD_REF"):
        return a["bridge_type"] == "STANDARD_REF"
    return len(a["bridge_norm"]) > len(b["bridge_norm"])


def _main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    facts = json.loads(open(argv[0], encoding="utf-8").read())
    if isinstance(facts, dict) and "facts" in facts:
        facts = facts["facts"]
    bidx = json.loads(open(argv[1], encoding="utf-8").read())
    res = build_bridge_pairs(facts, bidx)
    s = res["stats"]
    print(f"candidates={s['candidate_pairs']}  drop_missing={s['dropped_bridge_missing']}  "
          f"drop_dup={s['dropped_duplicate']}  VERIFIED={s['verified_pairs']}\n")
    by_id = {_get(f, 'fact_id', 'id'): f for f in facts}
    from collections import defaultdict
    by_doc = defaultdict(list)
    for p in res["pairs"]:
        by_doc[p["doc"]].append(p)
    for doc in sorted(by_doc, key=lambda d: -len(by_doc[d])):
        ps = by_doc[doc]
        print(f"## {doc}  ({len(ps)} verified pairs)")
        for p in ps[:4]:
            ta = _get(by_id[p["fact_a"]], "canonical_form", "text", default="")[:90]
            tb = _get(by_id[p["fact_b"]], "canonical_form", "text", default="")[:90]
            print(f"  [{p['bridge_type']}] bridge='{p['bridge_entity']}' pages={p['pages']} jac={p['content_jaccard']}")
            print(f"      A: {ta}")
            print(f"      B: {tb}")
        print()
    if len(argv) > 2:
        json.dump(res, open(argv[2], "w", encoding="utf-8"), indent=2)
        print(f"wrote {argv[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
