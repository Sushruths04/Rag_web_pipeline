"""Upgrade an existing GT JSONL's facts into Semantic Fact Units (SFUs).

Takes the existing facts (from V11 strict GT or earlier) and adds:
- ``canonical_form``: NLI-guarded self-contained rewrite
- ``self_containment_score``: 0..1 from the LLM-as-judge scorer
- ``raw_text``: preserves the original verbatim fact text

The semantic of ``text`` is preserved (used by all downstream matching code).
This is the MVP path — it doesn't rebuild the GT, it enriches it in place so
Phase 2 (CGA) and Phase 5 (full re-run) can compose gold answers from
canonical forms.

Usage:

  $env:PYTHONPATH = "src"
  python -m rag_gt.cli.upgrade_gt_to_sfu `
    --gt data/gt/reinforcement_qa_source_v11_multihop_strict_20260514.jsonl `
    --chunks-cache data/cache/chunks_rl_v11_char512_20260514.jsonl `
    --output data/gt/reinforcement_qa_source_v12_sfu_strict_20260516.jsonl `
    --max-context-chars 2000

`--chunks-cache` is required: the upgrader passes each fact's supporting
chunk text as ``context`` to the canonical-form LLM so it can resolve
discourse markers ("Thus", "It", "denoted X") to their antecedents.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import List

from loguru import logger

from rag_gt.comparison.chunk_resolver import ChunkResolver
from rag_gt.core.llm import get_llm
from rag_gt.core.types import Fact, QuestionGT
from rag_gt.facts.semantic_extraction import (
    MIN_SELF_CONTAINMENT_THRESHOLD,
    upgrade_fact_to_sfu,
)
from rag_gt.storage.gt_io import load_gt, save_gt


def _context_for_fact(fact: Fact, resolver: ChunkResolver, max_chars: int) -> str:
    """Concatenate the text of each chunk this fact's spans point at."""
    seen: set[str] = set()
    parts: List[str] = []
    for span in fact.supporting_spans:
        # Prefer the original chunk_id (V10-anchored), then any chunks the
        # active profile says overlap the span.
        cids: List[str] = []
        if span.chunk_id:
            cids.append(span.chunk_id)
        cids.extend(resolver.chunks_for_span(span))
        for cid in cids:
            if cid in seen:
                continue
            seen.add(cid)
            rec = resolver.record(cid)
            if rec is None:
                continue
            parts.append(rec.text)
            if sum(len(p) for p in parts) >= max_chars:
                break
        if sum(len(p) for p in parts) >= max_chars:
            break
    joined = "\n\n".join(parts).strip()
    return joined[:max_chars]


def run_upgrade(
    gt_path: Path,
    chunks_cache: Path,
    output_path: Path,
    *,
    limit: int | None = None,
    max_context_chars: int = 2000,
    min_self_containment: float = MIN_SELF_CONTAINMENT_THRESHOLD,
) -> int:
    questions = load_gt(gt_path.stem, in_dir=gt_path.parent or None)
    if limit is not None:
        questions = questions[:limit]
    if not questions:
        raise RuntimeError(f"No questions loaded from {gt_path}")

    resolver = ChunkResolver.from_cache(chunks_cache)
    coverage = resolver.verify_coverage(questions)
    print(
        f"[upgrade] chunks cache covers {coverage.found}/{coverage.requested} "
        f"required chunk_ids ({coverage.coverage:.1%})"
    )

    # Use the GT-grade LLM (gpt-oss-120b) for the canonical-form rewrite.
    # Reasoning is helpful here — bad rewrites are caught by the NLI guard.
    gt_llm = get_llm("gt")

    # Lazy-load the NLI model from validation; nli_batch handles loading on
    # first use.
    nli_model = True  # truthy placeholder; nli_batch loads the model itself

    seen_facts: set[str] = set()
    upgraded_count = 0
    nli_pass_count = 0
    weak_self_containment_count = 0
    score_distribution: Counter = Counter()
    t0 = time.time()

    for q_idx, q in enumerate(questions, start=1):
        for f in q.required_facts:
            if f.fact_id in seen_facts:
                continue
            seen_facts.add(f.fact_id)

            context = _context_for_fact(f, resolver, max_context_chars)
            try:
                upgrade_fact_to_sfu(
                    f,
                    context,
                    gt_llm,
                    nli_model=nli_model,
                    min_self_containment=min_self_containment,
                )
            except Exception as e:
                logger.warning(
                    f"[upgrade] fact={f.fact_id} failed: {type(e).__name__}: {e}"
                )
                continue
            upgraded_count += 1
            if f.canonical_form and f.canonical_form != f.raw_text:
                nli_pass_count += 1
            score_bucket = round(f.self_containment_score, 1)
            score_distribution[score_bucket] += 1
            if f.self_containment_score < min_self_containment:
                weak_self_containment_count += 1

        elapsed = time.time() - t0
        print(
            f"[upgrade] [{q_idx}/{len(questions)}] q={q.q_id} "
            f"facts_upgraded={upgraded_count} elapsed={elapsed:.1f}s"
        )

    save_gt(questions, output_path.stem, out_dir=output_path.parent or None)

    elapsed = time.time() - t0
    print("=" * 60)
    print("  SFU UPGRADE SUMMARY")
    print("=" * 60)
    print(f"  Facts upgraded         : {upgraded_count}")
    print(f"  Canonical rewrites     : {nli_pass_count} ({nli_pass_count / max(upgraded_count, 1):.1%})")
    print(f"  Weak self-containment  : {weak_self_containment_count} (score < {min_self_containment})")
    print(f"  Score distribution     : {dict(sorted(score_distribution.items()))}")
    print(f"  Wall time              : {elapsed / 60:.2f} min")
    print(f"  Output                 : {output_path}")
    print("=" * 60)

    summary = {
        "input_gt": str(gt_path),
        "output_gt": str(output_path),
        "facts_upgraded": upgraded_count,
        "canonical_rewrites": nli_pass_count,
        "weak_self_containment": weak_self_containment_count,
        "min_self_containment_threshold": min_self_containment,
        "score_distribution": {str(k): v for k, v in sorted(score_distribution.items())},
        "wall_time_seconds": elapsed,
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".upgrade_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Summary JSON           : {summary_path}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--chunks-cache", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--limit", type=int, default=None, help="Process only first N questions (debug).")
    p.add_argument(
        "--max-context-chars",
        type=int,
        default=2000,
        help="Max chunk-text length passed to the canonical-form LLM.",
    )
    p.add_argument(
        "--min-self-containment",
        type=float,
        default=MIN_SELF_CONTAINMENT_THRESHOLD,
    )
    args = p.parse_args()
    raise SystemExit(
        run_upgrade(
            args.gt,
            args.chunks_cache,
            args.output,
            limit=args.limit,
            max_context_chars=args.max_context_chars,
            min_self_containment=args.min_self_containment,
        )
    )


if __name__ == "__main__":
    main()
