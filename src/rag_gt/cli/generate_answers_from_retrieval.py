"""Generate real answer logs from retrieval outputs and cached chunk text.

`python -m rag_gt.cli.generate_answers_from_retrieval`

Inputs:
  --gt           GT JSONL file
  --retrieval    retrieval_logs.jsonl
  --chunks-cache chunk cache JSONL
  --output       answer_logs.jsonl

The CLI uses the existing `answer` LLM role and answers each question from the
retrieved chunk texts only. This fills the missing gap between retrieval-only
evaluation and the end-to-end `rag-gt-compare` harness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from rag_gt.comparison.chunk_resolver import ChunkResolver
from rag_gt.core.llm import get_llm
from rag_gt.core.types import AnswerLog, Fact, RetrievalLog
from rag_gt.generation.answers import ABSTENTION_TEXT, generate_answer
from rag_gt.storage.gt_io import load_gt


def _read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load_retrieval(path: Path) -> Dict[str, RetrievalLog]:
    return {
        row["q_id"]: RetrievalLog(
            q_id=row["q_id"],
            retrieved_chunk_ids=row.get("retrieved_chunk_ids", []),
        )
        for row in _read_jsonl(path)
    }


def _facts_from_contexts(
    q_id: str,
    retrieved_chunk_ids: Iterable[str],
    resolver: ChunkResolver,
    max_context_chunks: int,
    max_chars_per_chunk: int,
) -> List[Fact]:
    facts: List[Fact] = []
    for i, cid in enumerate(list(retrieved_chunk_ids)[:max_context_chunks], start=1):
        rec = resolver.record(cid)
        if rec is None:
            continue
        text = (rec.text or "").strip()
        if len(text) < 10:
            continue
        facts.append(
            Fact(
                fact_id=f"{q_id}_ctx{i:03d}",
                text=text[:max_chars_per_chunk],
                role="example",
            )
        )
    return facts


def run_generation(
    gt_path: Path,
    retrieval_path: Path,
    chunks_cache: Path,
    output_path: Path,
    *,
    limit: int | None = None,
    max_context_chunks: int = 8,
    max_chars_per_chunk: int = 1200,
) -> int:
    questions = load_gt(gt_path.stem, in_dir=gt_path.parent or None)
    q_map = {q.q_id: q for q in questions}
    retrieval = _load_retrieval(retrieval_path)
    resolver = ChunkResolver.from_cache(chunks_cache)
    llm = get_llm("answer")

    matched_ids = [qid for qid in q_map if qid in retrieval]
    if limit is not None:
        matched_ids = matched_ids[:limit]
    if not matched_ids:
        raise RuntimeError("No matching q_ids across GT and retrieval logs.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[AnswerLog] = []
    for idx, qid in enumerate(matched_ids, start=1):
        q = q_map[qid]
        ret = retrieval[qid]
        context_facts = _facts_from_contexts(
            qid,
            ret.retrieved_chunk_ids,
            resolver,
            max_context_chunks=max_context_chunks,
            max_chars_per_chunk=max_chars_per_chunk,
        )
        if not context_facts:
            rows.append(AnswerLog(q_id=qid, predicted_answer="", abstained=True))
            print(f"[{idx}/{len(matched_ids)}] {qid}: abstained (no cached retrieved contexts)")
            continue
        answer = generate_answer(q.question, context_facts, llm)
        abstained = answer.strip() == ABSTENTION_TEXT
        rows.append(AnswerLog(q_id=qid, predicted_answer=answer, abstained=abstained))
        status = "abstained" if abstained else "answered"
        print(f"[{idx}/{len(matched_ids)}] {qid}: {status}")

    with output_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(
                json.dumps(
                    {
                        "q_id": row.q_id,
                        "predicted_answer": row.predicted_answer,
                        "abstained": row.abstained,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    answered = sum(1 for r in rows if not r.abstained and r.predicted_answer.strip())
    abstained = sum(1 for r in rows if r.abstained)
    print(
        f"[generate_answers_from_retrieval] wrote {len(rows)} rows to {output_path} "
        f"(answered={answered}, abstained={abstained})"
    )
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--retrieval", required=True, type=Path)
    p.add_argument("--chunks-cache", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-context-chunks", type=int, default=8)
    p.add_argument("--max-chars-per-chunk", type=int, default=1200)
    args = p.parse_args()
    raise SystemExit(
        run_generation(
            args.gt,
            args.retrieval,
            args.chunks_cache,
            args.output,
            limit=args.limit,
            max_context_chunks=args.max_context_chunks,
            max_chars_per_chunk=args.max_chars_per_chunk,
        )
    )


if __name__ == "__main__":
    main()
