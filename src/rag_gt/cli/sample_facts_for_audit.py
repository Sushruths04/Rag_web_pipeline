"""`python -m rag_gt.cli.sample_facts_for_audit` -- sample facts for human audit.

Random-samples N facts from `data/cache/facts.jsonl` (built by
`cli/build_fact_store.py`), resolves the supporting chunk text via
`ChunkResolver`, and writes a CSV the auditor fills in. Pair with
`cli/score_audit.py` to summarise the filled CSV.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List

from rag_gt.comparison.chunk_resolver import ChunkResolver
from rag_gt.comparison.fact_store import FactStore


def main() -> None:
    p = argparse.ArgumentParser(
        prog="rag-gt-sample-facts-for-audit",
        description="Sample N facts → CSV for human auditing.",
    )
    p.add_argument("--facts", default="data/cache/facts.jsonl")
    p.add_argument("--chunks-cache", default="data/cache/chunks.jsonl")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="data/eval_results/fact_audit/audit.csv")
    args = p.parse_args()

    store = FactStore.from_cache(args.facts)
    resolver = ChunkResolver.from_cache(args.chunks_cache)
    fact_ids: List[str] = sorted(store)
    if not fact_ids:
        raise SystemExit("No facts in store; run cli/build_fact_store first.")

    rng = random.Random(args.seed)
    n = min(args.n, len(fact_ids))
    sample = rng.sample(fact_ids, n)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "fact_id", "doc_id", "role", "fact_text",
            "supporting_chunk_id", "supporting_chunk_text",
            "well_formed?", "span_correct?", "notes",
        ])
        for fid in sample:
            f = store.get(fid)
            sup_id = f.supporting_spans[0].chunk_id if f.supporting_spans else ""
            sup_text = ""
            if sup_id:
                try:
                    sup_text = resolver.get(sup_id)
                except KeyError:
                    sup_text = "(chunk not in cache)"
            w.writerow([
                f.fact_id,
                (f.supporting_spans[0].doc_id if f.supporting_spans else ""),
                f.role,
                f.text,
                sup_id,
                sup_text,
                "", "", "",
            ])
    print(f"[sample_facts_for_audit] wrote {n} rows -> {out_path}")
    print("Open the CSV, fill `well_formed?` and `span_correct?` columns "
          "with 1/0, then run `cli/score_audit`.")


if __name__ == "__main__":
    main()
