"""Measure what question forms the current prompt actually produces.

A gate, not a benchmark: run before spending anything on GT generation, to
confirm the question-form affordance fix genuinely diversifies stems rather
than defaulting everything to "What".

    python scripts/probe_question_forms.py --n 30

Reads facts straight from the stage-3 chunk caches (no pipeline run, no
extraction cost) and generates one question per fact with the anchor model.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

DOCS = ["din_iso_6507_1", "din_iso_3452_1"]
CACHE = "data/reruns_postfix/{doc}/checkpoints/s3_chunk_cache/*.json"
ANCHOR_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
SEED = 42


def load_facts():
    from rag_gt.core.types import Fact, Span

    out = []
    for doc in DOCS:
        for p in sorted(glob.glob(CACHE.format(doc=doc))):
            for d in json.load(open(p, encoding="utf-8")):
                out.append((doc, Fact(
                    fact_id=d["fact_id"], text=d["text"],
                    raw_text=d.get("raw_text") or d["text"],
                    canonical_form=d.get("canonical_form") or d["text"],
                    role=d.get("role", "descriptive"), weight=d.get("weight", 0.5),
                    self_containment_score=d.get("self_containment_score", 1.0),
                    supporting_spans=[Span.from_any(s) for s in d.get("spans", [])],
                )))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    from rag_gt.core.llm import APILLM
    from rag_gt.generation.questions import generate_question, question_affordances

    llm = APILLM(
        base_url=os.getenv("API_BASE_URL", "").strip().strip('"'),
        api_key=os.getenv("API_KEY", "").strip().strip('"'),
        model=ANCHOR_MODEL,
    )

    facts = load_facts()
    rng = random.Random(SEED)
    rng.shuffle(facts)
    sample = facts[:args.n]
    print(f"probing {len(sample)} facts with {ANCHOR_MODEL}\n")

    def one(item):
        doc, f = item
        try:
            q = generate_question([f], llm)
        except Exception as e:
            q = f"<ERROR {type(e).__name__}: {e}>"
        return doc, f, q

    with ThreadPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(one, sample))

    stems = collections.Counter()
    by_hint = collections.Counter()
    for doc, f, q in rows:
        stem = (q or "").strip().split()[0].strip('",').capitalize() if q else "<empty>"
        stems[stem] += 1
        aff = question_affordances(f.canonical_form or f.text)
        by_hint[("hint" if aff else "NO-hint", stem)] += 1
        print(f"[{f.role:11s}] aff={','.join(aff) or '-':<22s} {q}")

    total = sum(stems.values()) or 1
    print(f"\n{'='*70}\nSTEM DISTRIBUTION ({total} questions)")
    for s, c in stems.most_common():
        print(f"  {s:<12s} {c:3d}  {c/total*100:5.1f}%  {'#' * c}")

    what = stems.get("What", 0) / total * 100
    print(f"\n'What' share: {what:.1f}%")

    print("\nstem by whether the prompt supplied an affordance hint:")
    for (h, s), c in sorted(by_hint.items()):
        print(f"  {h:<8s} {s:<12s} {c}")

    print(f"\nusage: {llm.usage_totals}")
    verdict = "PASS" if what <= 60 else "FAIL"
    print(f"\nVERDICT: {verdict} — 'What' at {what:.1f}% (gate: <=60%)")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
