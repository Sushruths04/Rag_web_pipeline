"""Compress a broad retrieval candidate set into a smaller final evidence set.

Use case for source-anchored RAG_GT:

  1. Retrieve broadly for recall, e.g. FactIdx top-30.
  2. Rerank those candidate chunks against the question.
  3. Keep top-5/top-8/top-10 as the final answer/evaluation context.

The output is the same retrieval_logs.jsonl schema used by the evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _norm_minmax(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return np.ones_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def main() -> None:
    p = argparse.ArgumentParser(
        prog="rag-gt-compress-retrieval",
        description=__doc__,
    )
    p.add_argument("--gt", required=True, help="GT JSONL with q_id/question rows.")
    p.add_argument(
        "--candidates",
        required=True,
        help="Broad retrieval log, e.g. FactIdx top-30.",
    )
    p.add_argument(
        "--chunks",
        required=True,
        help="Chunk cache JSONL with chunk_id/text rows.",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument(
        "--method",
        choices=["cross_encoder", "blend"],
        default="blend",
        help=(
            "cross_encoder = pure reranker score; blend = reranker score plus "
            "a small original-rank prior to preserve FactIdx evidence ordering."
        ),
    )
    p.add_argument("--reranker_model", default="BAAI/bge-reranker-base")
    p.add_argument(
        "--original_weight",
        type=float,
        default=0.15,
        help="Blend weight for original candidate rank when --method blend.",
    )
    args = p.parse_args()

    gt = _load_jsonl(Path(args.gt))
    q_by_id = {row["q_id"]: row.get("question", "") for row in gt}
    candidate_rows = _load_jsonl(Path(args.candidates))

    chunks = _load_jsonl(Path(args.chunks))
    text_by_id: Dict[str, str] = {}
    for c in chunks:
        cid = c.get("chunk_id") or c.get("id")
        if cid:
            text_by_id[str(cid)] = str(c.get("text", "") or "")

    from sentence_transformers import CrossEncoder

    print(f"Loading cross-encoder reranker: {args.reranker_model}")
    reranker = CrossEncoder(args.reranker_model)

    print(
        f"Compressing {len(candidate_rows)} questions "
        f"(method={args.method}, top_k={args.top_k})"
    )
    out_rows: List[dict] = []
    missing_text = 0
    for row in candidate_rows:
        qid = row["q_id"]
        question = q_by_id.get(qid, "")
        cand_ids = [str(cid) for cid in row.get("retrieved_chunk_ids", [])]
        cand_ids = [cid for cid in cand_ids if cid in text_by_id]
        if not question or not cand_ids:
            out_rows.append({"q_id": qid, "retrieved_chunk_ids": cand_ids[: args.top_k]})
            continue
        missing_text += len(row.get("retrieved_chunk_ids", [])) - len(cand_ids)

        pairs = [(question, text_by_id[cid]) for cid in cand_ids]
        ce_scores = np.asarray(reranker.predict(pairs, show_progress_bar=False), dtype=float)
        if args.method == "blend":
            # Candidate order came from FactIdx. Keep a small prior so the
            # compressor does not throw away strong source-fact candidates when
            # the cross-encoder scores are close.
            rank_prior = np.asarray(
                [1.0 / (rank + 1) for rank in range(len(cand_ids))],
                dtype=float,
            )
            final_scores = _norm_minmax(ce_scores) + float(args.original_weight) * _norm_minmax(rank_prior)
        else:
            final_scores = ce_scores

        order = np.argsort(-final_scores)
        selected = [cand_ids[int(i)] for i in order[: args.top_k]]
        out_rows.append({"q_id": qid, "retrieved_chunk_ids": selected})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"written {len(out_rows)} compressed retrieval logs to {out}")
    if missing_text:
        print(f"warning: skipped {missing_text} candidate ids missing from chunk cache")


if __name__ == "__main__":
    main()
