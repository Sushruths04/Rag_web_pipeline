"""Post-processing step: apply P5 ARM to all strict rows in a v16 GT file.

ARM (Abstractive Reasoning Map) replaces the plain-LLM gold answer with a
decompose-then-synthesize answer that carries a per-step reasoning_trace.
This script is needed because the main generation pipeline writes gold answers
via generate_answer() (fast, simple) and ARM is applied as a separate pass.

Usage:
    python -m rag_gt.cli.apply_arm \\
        --gt data/gt/reinforcement_qa_source_v16_pass_gt_20260519.jsonl \\
        [--out data/gt/reinforcement_qa_source_v16_arm_20260519.jsonl] \\
        [--limit 20]

Output: a new JSONL with reasoning_trace populated for all strict rows.
Twin rows (abstention/counterfactual) are passed through unchanged.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import List  # noqa: F401

from loguru import logger

from rag_gt.core.llm import get_llm
from rag_gt.generation.cga import build_constructive_gold_answer
from rag_gt.storage.gt_io import load_gt, save_gt


def run_arm(
    gt_path: Path,
    output_path: Path,
    *,
    limit: int | None = None,
    arm_threshold: float = 0.55,
    arm_max_np_overlap: float = 0.35,
    arm_max_retries: int = 3,
) -> dict:
    questions = load_gt(gt_path.stem, in_dir=gt_path.parent)
    if limit is not None:
        questions = questions[:limit]
    if not questions:
        raise RuntimeError(f"No questions loaded from {gt_path}")

    llm = get_llm("answer")

    np_overlaps: List[float] = []
    nli_scores: List[float] = []
    arm_applied = 0
    arm_failed = 0
    t0 = time.time()

    for idx, q in enumerate(questions, start=1):
        if q.twin_type is not None:
            continue

        if not q.required_facts:
            logger.warning(f"[ARM] q={q.q_id} has no required_facts — skipping")
            continue

        try:
            result = build_constructive_gold_answer(
                q.question,
                q.required_facts,
                llm,
                decomposer_llm=llm,
                threshold=arm_threshold,
                max_retries=arm_max_retries,
                use_arm=True,
                tf_sfg_edges=q.tf_sfg_edges or [],
            )
            q.gold_answer = result.gold_answer
            q.reasoning_trace = result.reasoning_trace

            for step in result.reasoning_trace:
                if step.get("np_overlap") is not None:
                    np_overlaps.append(float(step["np_overlap"]))
                if step.get("nli_score") is not None:
                    nli_scores.append(float(step["nli_score"]))

            arm_applied += 1
            if idx % 10 == 0:
                logger.info(f"[ARM] {idx} rows processed, {arm_applied} ARM applied")

        except Exception as e:
            logger.warning(f"[ARM] q={q.q_id} ARM failed: {type(e).__name__}: {e}")
            arm_failed += 1

    elapsed = time.time() - t0
    save_gt(questions, output_path.stem, out_dir=output_path.parent)

    stats = {
        "total_rows": len(questions),
        "arm_applied": arm_applied,
        "arm_failed": arm_failed,
        "elapsed_s": round(elapsed, 1),
        "arm_steps": len(np_overlaps),
        "mean_np_overlap": round(statistics.mean(np_overlaps), 4) if np_overlaps else None,
        "std_np_overlap": round(statistics.stdev(np_overlaps), 4) if len(np_overlaps) > 1 else None,
        "mean_nli_score": round(statistics.mean(nli_scores), 4) if nli_scores else None,
        "pct_steps_pass_nli": round(
            sum(1 for s in nli_scores if s >= arm_threshold) / len(nli_scores) * 100, 1
        ) if nli_scores else None,
    }

    stats_path = output_path.with_suffix(".arm_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    logger.info(f"[ARM] Done. Stats: {stats}")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(prog="apply-arm")
    p.add_argument("--gt", required=True, help="v16 GT JSONL path")
    p.add_argument("--out", default=None, help="Output path (default: input.arm.jsonl)")
    p.add_argument("--limit", type=int, default=None, help="Process only first N strict rows")
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--max-np-overlap", type=float, default=0.35)
    p.add_argument("--max-retries", type=int, default=3)
    args = p.parse_args()

    gt_path = Path(args.gt)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = gt_path.with_suffix("").with_suffix(".arm.jsonl")

    stats = run_arm(
        gt_path,
        out_path,
        limit=args.limit,
        arm_threshold=args.threshold,
        arm_max_np_overlap=args.max_np_overlap,
        arm_max_retries=args.max_retries,
    )

    print("\n=== ARM Post-Processing Complete ===")
    print(f"ARM applied:      {stats['arm_applied']} rows")
    print(f"ARM failed:       {stats['arm_failed']} rows")
    print(f"Total steps:      {stats['arm_steps']}")
    print(f"Mean NP overlap:  {stats['mean_np_overlap']}  (target ≤ 0.35)")
    print(f"Std NP overlap:   {stats['std_np_overlap']}")
    print(f"Mean NLI score:   {stats['mean_nli_score']}  (target ≥ 0.55)")
    print(f"% steps pass NLI: {stats['pct_steps_pass_nli']}%")
    print(f"Elapsed:          {stats['elapsed_s']} s")
    print(f"Output:           {out_path}")
    print(f"Stats file:       {out_path.with_suffix('').with_suffix('.arm_stats.json')}")


if __name__ == "__main__":
    main()
