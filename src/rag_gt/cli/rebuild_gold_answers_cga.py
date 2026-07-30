"""Rebuild every gold_answer in a GT JSONL using Constructive Gold Answer (CGA).

Phase 2 + 5 of the SFU+CGA+SAE plan (plan v6). For each question:

  1. Read the chain of required_facts (SFUs).
  2. Call `build_constructive_gold_answer(question, chain_facts, ...)` which
     composes canonical_forms, paraphrases for fluency, decomposes into
     atomic clauses, and NLI-guards each clause against the SFUs.
  3. Replace `gold_answer` with the CGA result.
  4. Save the modified question, or drop it when --c1-hard-gate is set and
     at least one generated answer clause is not entailed by the SFUs.

Output: a new GT JSONL (does not mutate the input) plus a per-question
diagnostic JSON listing how many atomic clauses were grounded against the
SFUs for each question. By default, questions whose CGA result has
`all_clauses_pass = False` are kept but flagged. With `--c1-hard-gate`, they
are removed from the output GT and also written to a `.cga_drops.jsonl` file.

Best run after `rag_gt.cli.upgrade_gt_to_sfu` so each fact has
`canonical_form` populated. Falls back to `Fact.text` for any fact missing
the canonical form (works on legacy V11 GTs as a degraded baseline).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

from loguru import logger

from rag_gt.core.llm import get_llm
from rag_gt.core.types import QuestionGT
from rag_gt.generation.cga import build_constructive_gold_answer
from rag_gt.storage.gt_io import load_gt, save_gt


def run_rebuild(
    gt_path: Path,
    output_path: Path,
    *,
    limit: int | None = None,
    use_llm_decomposer: bool = True,
    c1_hard_gate: bool = False,
) -> int:
    questions = load_gt(gt_path.stem, in_dir=gt_path.parent or None)
    if limit is not None:
        questions = questions[:limit]
    if not questions:
        raise RuntimeError(f"No questions loaded from {gt_path}")

    # Same LLM (gpt-oss-120b) for paraphrase and decomposer. CoT in
    # decomposer output is fine because we parse JSON anyway.
    paraphrase_llm = get_llm("gt")
    decomposer_llm = get_llm("gt") if use_llm_decomposer else None

    diagnostics: List[dict] = []
    passed_count = 0
    # C1: track which q_ids fully pass CGA so we can hard-gate on them.
    passed_q_ids: set = set()
    t0 = time.time()

    for q_idx, q in enumerate(questions, start=1):
        try:
            result = build_constructive_gold_answer(
                q.question,
                q.required_facts,
                paraphrase_llm,
                decomposer_llm=decomposer_llm,
            )
        except Exception as e:
            logger.warning(
                f"[CGA-rebuild] q={q.q_id} failed: {type(e).__name__}: {e}; "
                f"keeping legacy gold_answer"
            )
            diagnostics.append(
                {
                    "q_id": q.q_id,
                    "all_clauses_pass": False,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            continue

        q.gold_answer = result.gold_answer
        if result.all_clauses_pass:
            passed_count += 1
            passed_q_ids.add(q.q_id)

        diagnostics.append(
            {
                "q_id": q.q_id,
                "all_clauses_pass": result.all_clauses_pass,
                "grounded_clauses": result.grounded_clause_count,
                "total_clauses": result.total_clause_count,
                "retries": result.retries,
                "construction_method": result.construction_method,
                "atomic_clauses": [
                    {
                        "clause": g.clause,
                        "passes": g.passes,
                        "best_score": round(g.best_score, 3),
                        "grounded_sfu_index": g.grounded_sfu_index,
                    }
                    for g in result.clause_grounding
                ],
            }
        )

        elapsed = time.time() - t0
        status = "PASS" if result.all_clauses_pass else "PARTIAL"
        print(
            f"[CGA-rebuild] [{q_idx}/{len(questions)}] q={q.q_id} [{status}] "
            f"{result.grounded_clause_count}/{result.total_clause_count} clauses "
            f"retries={result.retries} elapsed={elapsed:.1f}s"
        )

    # C1 hard-gate: drop questions where all_clauses_pass=False.
    dropped_count = 0
    drop_path = output_path.with_suffix(output_path.suffix + ".cga_drops.jsonl")
    if c1_hard_gate:
        n_before = len(questions)
        questions = [q for q in questions if q.q_id in passed_q_ids]
        dropped_count = n_before - len(questions)
        dropped_diags = [
            d for d in diagnostics if not bool(d.get("all_clauses_pass", False))
        ]
        with open(drop_path, "w", encoding="utf-8", newline="\n") as fh:
            for d in dropped_diags:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        if dropped_count:
            logger.info(f"[C1] hard-gate dropped {dropped_count} questions (all_clauses_pass=False)")

    save_gt(questions, output_path.stem, out_dir=output_path.parent or None)
    diag_path = output_path.with_suffix(output_path.suffix + ".cga_diagnostics.json")
    diag_path.write_text(
        json.dumps(
            {
                "input_gt": str(gt_path),
                "output_gt": str(output_path),
                "n_questions_input": len(questions) + dropped_count,
                "n_questions_output": len(questions),
                "c1_hard_gate": c1_hard_gate,
                "c1_dropped": dropped_count,
                "all_clauses_pass_count": passed_count,
                "all_clauses_pass_rate": passed_count / max(len(questions) + dropped_count, 1),
                "wall_time_seconds": time.time() - t0,
                "per_question": diagnostics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    elapsed = time.time() - t0
    n_saved = len(questions)
    print("=" * 60)
    print("  CGA REBUILD SUMMARY")
    print("=" * 60)
    print(f"  Questions processed     : {n_saved + dropped_count}")
    print(f"  All-clauses-pass        : {passed_count} ({passed_count/max(n_saved + dropped_count,1):.1%})")
    if c1_hard_gate:
        print(f"  C1 hard-gate dropped    : {dropped_count}")
        print(f"  Questions saved (C1)    : {n_saved}")
        print(f"  C1 drops JSONL          : {drop_path}")
    print(f"  Wall time               : {elapsed/60:.2f} min")
    print(f"  Output GT               : {output_path}")
    print(f"  Diagnostics JSON        : {diag_path}")
    print("=" * 60)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--no-llm-decomposer",
        action="store_true",
        help="Use the heuristic atomic-clause splitter instead of the LLM decomposer (cheaper).",
    )
    p.add_argument(
        "--c1-hard-gate",
        action="store_true",
        help="C1: drop questions where all_clauses_pass=False from the output GT.",
    )
    args = p.parse_args()
    raise SystemExit(
        run_rebuild(
            args.gt,
            args.output,
            limit=args.limit,
            use_llm_decomposer=not args.no_llm_decomposer,
            c1_hard_gate=args.c1_hard_gate,
        )
    )


if __name__ == "__main__":
    main()
