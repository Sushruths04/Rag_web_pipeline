"""`python -m rag_gt.cli.build_fact_store` -- build/extend the fact store.

Two build modes:

  --from-gt PATH        Harvest facts already embedded inline in a new-format
                        GT JSONL. Fast, no PDF processing.

  --from-source DIR     Re-run the pipeline's facts step over every PDF/DOCX
                        in DIR. Required for old-format GT (only fact_ids,
                        no embedded fact text). Slow but thorough.

Both modes can be combined; later sources overwrite earlier entries on
fact_id collision. Output is a JSONL keyed by fact_id, suitable for
`FactStore.from_cache(...)`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_gt.comparison.fact_store import FactStore


def main() -> None:
    p = argparse.ArgumentParser(
        prog="rag-gt-build-fact-store",
        description="Build the fact_id -> Fact JSONL cache.",
    )
    p.add_argument(
        "--from-gt",
        action="append",
        default=[],
        help="GT JSONL with embedded `required_facts` (repeatable).",
    )
    p.add_argument(
        "--from-source",
        action="append",
        default=[],
        help="Source-document directory; re-runs facts extraction (repeatable).",
    )
    p.add_argument(
        "--output",
        default="data/cache/facts.jsonl",
        help="Output JSONL (default: data/cache/facts.jsonl).",
    )
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--chunk_overlap", type=int, default=64)
    args = p.parse_args()

    if not args.from_gt and not args.from_source:
        sys.stderr.write(
            "build_fact_store: pass at least one of --from-gt or --from-source.\n"
        )
        sys.exit(2)

    store = FactStore()

    for gt_path in args.from_gt:
        sub = FactStore.from_gt_inline(gt_path)
        print(f"[build_fact_store] from GT {gt_path}: +{len(sub)} facts")
        store = store.merge(sub)

    for src_dir in args.from_source:
        sub = FactStore.build_from_source(
            src_dir, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )
        print(f"[build_fact_store] from source {src_dir}: +{len(sub)} facts")
        store = store.merge(sub)

    out = Path(args.output)
    store.save(out)
    print(f"[build_fact_store] wrote {len(store)} facts -> {out}")


if __name__ == "__main__":
    main()
