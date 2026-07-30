"""`python -m rag_gt.cli.evaluate` -- Phase 2 evaluation CLI.

Phase 4 (plan v5) note: `_fact_span_recall` and `_fact_span_precision` now
accept an optional `ChunkResolver`. When the resolver is provided, GT
`supporting_spans` whose `chunk_id` does not directly match the retrieval log
(because the retrieval log uses a different chunk profile than the GT was
generated against — the V11 dynamic-chunking case) are projected into the
active chunk profile via source-span overlap. Without this, V11 retrieval
logs would always score ~0 against V10-generated GT, even when retrieval
actually covers the required source spans.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from loguru import logger
from rapidfuzz import fuzz
from tqdm import tqdm

from rag_gt.core.models import MM, NLI_LABEL_INDEX
from rag_gt.core.types import AnswerLog, Fact, QuestionGT, RetrievalLog, Span
from rag_gt.storage.gt_io import load_gt
from rag_gt.validation.nli_check import nli_batch, nli_entailment

if TYPE_CHECKING:
    from rag_gt.comparison.chunk_resolver import ChunkResolver

METRIC_KEYS = [
    "fact_span_recall",
    "fact_span_precision",
    "kpr",
    "fact_precision",
    "faithfulness",
    "contradiction_rate",
    "abstention_correct",
    "hallucination_flag",
]


def _spans(facts: List[Fact]) -> List[Span]:
    out: List[Span] = []
    for f in facts:
        out.extend(f.supporting_spans)
    return out


def _expected_chunk_ids_for_span(
    span: Span, resolver: Optional["ChunkResolver"]
) -> List[str]:
    """Return the chunk IDs in the resolver's active profile that overlap this span.

    Falls back to `[span.chunk_id]` when the resolver isn't available or the
    span has no character offsets to project from. This keeps the legacy
    chunk-id-equality behaviour intact when --chunks-cache isn't passed.

    NOTE: The caller (the comparator's `_run_rag_gt` path) normalises chunk
    IDs in the GT and the retrieval log via `_normalized_key`, which strips
    leading zeros from `_c\\d+` suffixes. Without also normalising the IDs
    returned here, projection silently fails: the resolver yields `c000711`
    but the retrieval set has the normalised form `c711`, so set membership
    is always false. This helper now returns BOTH the verbatim resolver
    output and its normalised form, so the downstream `set & set` check
    works regardless of which form the caller normalises to.
    """
    from rag_gt.comparison.chunk_resolver import _normalized_key

    if resolver is None or span.char_start is None or span.char_end is None:
        out: List[str] = []
        if span.chunk_id:
            out.append(span.chunk_id)
            n = _normalized_key(span.chunk_id)
            if n != span.chunk_id:
                out.append(n)
        return out

    projected = list(resolver.chunks_for_span(span))
    if span.chunk_id and span.chunk_id not in projected:
        projected.insert(0, span.chunk_id)

    # Append the normalised form for each projected id (dedup-preserving).
    seen = set(projected)
    extra: List[str] = []
    for cid in projected:
        n = _normalized_key(cid)
        if n not in seen:
            extra.append(n)
            seen.add(n)
    return projected + extra


def _fact_span_recall(
    retrieved_chunks: List[str],
    facts: List[Fact],
    resolver: Optional["ChunkResolver"] = None,
) -> float:
    spans = _spans(facts)
    if not spans:
        return 1.0
    retrieved_set = set(retrieved_chunks)
    matched = 0
    for s in spans:
        candidates = _expected_chunk_ids_for_span(s, resolver)
        if any(cid in retrieved_set for cid in candidates):
            matched += 1
    return matched / len(spans)


def _fact_span_precision(
    retrieved_chunks: List[str],
    facts: List[Fact],
    resolver: Optional["ChunkResolver"] = None,
) -> float:
    if not retrieved_chunks:
        return 0.0
    req_chunks: set[str] = set()
    for s in _spans(facts):
        for cid in _expected_chunk_ids_for_span(s, resolver):
            if cid:
                req_chunks.add(cid)
    if not req_chunks:
        # Symmetric with `_fact_span_recall`'s 1.0 default — when no required
        # chunks exist there is no precision claim to make. Return NaN so the
        # aggregator can skip cleanly; downstream `mean` filters NaN below.
        return float("nan")
    retrieved_set = set(retrieved_chunks)
    matched = len(retrieved_set & req_chunks)
    return matched / len(retrieved_set)


def _kpr(pred: str, gold: str) -> float:
    if not pred.strip() or not gold.strip():
        return 0.0
    return fuzz.ratio(pred.lower(), gold.lower()) / 100.0


def _contradiction_index() -> int:
    """Resolve the contradiction label index. Loads NLI if needed."""
    if "contradiction" not in NLI_LABEL_INDEX:
        MM.load_nli()
    idx = NLI_LABEL_INDEX.get("contradiction")
    if idx is None:
        raise RuntimeError(
            "NLI label index for 'contradiction' is unknown. Check core/models.py."
        )
    return int(idx)


def _contradictions_batch(pairs: List[Tuple[str, str]]) -> List[float]:
    """Return contradiction probabilities for each (premise, hypothesis) pair.

    Uses a fresh batch through the underlying NLI cross-encoder; the NLI cache
    keys on entailment and is not reused for contradiction probes."""
    if not pairs:
        return []
    model = MM.get_nli()
    contra_idx = _contradiction_index()
    scores = model.predict(pairs, apply_softmax=True)
    return [float(s[contra_idx]) for s in scores]


def evaluate_answer(
    q_gt: QuestionGT,
    retrieval: RetrievalLog,
    answer_log: AnswerLog,
    resolver: Optional["ChunkResolver"] = None,
) -> Dict[str, float]:
    required = q_gt.required_facts
    if not required:
        raise ValueError(
            f"GT question '{q_gt.q_id}' is missing required_facts. "
            "Regenerate GT with the current pipeline."
        )
    if any(not f.supporting_spans for f in required):
        raise ValueError(
            f"GT question '{q_gt.q_id}' contains required_facts without supporting_spans."
        )
    fsr = _fact_span_recall(retrieval.retrieved_chunk_ids, required, resolver)
    fsp = _fact_span_precision(retrieval.retrieved_chunk_ids, required, resolver)
    kpr = _kpr(answer_log.predicted_answer, q_gt.gold_answer)

    pred = answer_log.predicted_answer
    gold = q_gt.gold_answer

    # NLI-based metrics: still per-question here; the corpus-level run uses the
    # batched path for speed.
    context = " ".join(f.text for f in required)
    fp = nli_entailment(context, pred)
    if answer_log.abstained:
        # Abstention-aware (Attempt 5): an "Insufficient information" reply has
        # no propositional content to NLI-grade against the gold answer.
        # Counting it as low-faithfulness / high-contradiction conflates two
        # different failure modes (over-abstention vs hallucination). Emit NaN
        # so the corpus aggregator skips abstained rows for these two metrics.
        # `abstention_correct` and `hallucination_flag` still fire normally.
        faith = float("nan")
        contra = float("nan")
    else:
        faith = nli_entailment(gold, pred)
        contra = _contradictions_batch([(gold, pred)])[0]

    # Symmetric abstention reward: we credit "abstained when retrieval was
    # poor" AND "answered when retrieval was good".
    abst_ok = 1.0 if (answer_log.abstained == (fsr < 0.5)) else 0.0
    hallucination = (
        1.0
        if (not answer_log.abstained and faith == faith and faith < 0.3)
        else 0.0
    )
    return {
        "fact_span_recall": fsr,
        "fact_span_precision": fsp,
        "kpr": kpr,
        "fact_precision": fp,
        "faithfulness": faith,
        "contradiction_rate": contra,
        "abstention_correct": abst_ok,
        "hallucination_flag": hallucination,
    }


def _evaluate_corpus_batched(
    q_map: Dict[str, QuestionGT],
    ret_raw: Dict[str, RetrievalLog],
    ans_raw: Dict[str, AnswerLog],
    resolver: Optional["ChunkResolver"] = None,
) -> List[dict]:
    """Compute all metrics with batched NLI calls — much faster than per-Q.

    When `resolver` is provided, fact-span metrics use source-span projection
    into the resolver's active chunk profile, so the V11 dynamic-chunking
    case (retrieval log in a different profile than the GT's original chunks)
    measures correctly. Without it, behaviour is identical to the V10
    chunk-id-equality path.
    """
    matched_ids = [qid for qid in q_map if qid in ret_raw and qid in ans_raw]
    n = len(matched_ids)
    if n == 0:
        return []

    fp_pairs: List[Tuple[str, str]] = []
    # Abstention-aware (Attempt 5): we only batch faith/contra pairs for
    # non-abstained rows. Abstained rows get NaN for both metrics so the
    # corpus aggregator skips them (otherwise an "Insufficient information"
    # reply is double-counted as a contradiction against positive gold and
    # collapses both means even though no claim was actually made).
    faith_pairs: List[Tuple[str, str]] = []
    contra_pairs: List[Tuple[str, str]] = []
    faith_indices: List[int] = []

    base_rows: List[dict] = []
    for qid in matched_ids:
        q_gt = q_map[qid]
        ret = ret_raw[qid]
        ans = ans_raw[qid]
        required = q_gt.required_facts
        if not required:
            raise ValueError(
                f"GT question '{qid}' is missing required_facts. Regenerate GT."
            )
        if any(not f.supporting_spans for f in required):
            raise ValueError(
                f"GT question '{qid}' contains required_facts without supporting_spans."
            )

        fsr = _fact_span_recall(ret.retrieved_chunk_ids, required, resolver)
        fsp = _fact_span_precision(ret.retrieved_chunk_ids, required, resolver)
        kpr = _kpr(ans.predicted_answer, q_gt.gold_answer)
        context = " ".join(f.text for f in required)
        fp_pairs.append((context, ans.predicted_answer))

        row_idx = len(base_rows)
        if not ans.abstained:
            faith_pairs.append((q_gt.gold_answer, ans.predicted_answer))
            contra_pairs.append((q_gt.gold_answer, ans.predicted_answer))
            faith_indices.append(row_idx)

        base_rows.append(
            {
                "q_id": qid,
                "fsr": fsr,
                "fsp": fsp,
                "kpr": kpr,
                "abstained": ans.abstained,
            }
        )

    fp_scores = nli_batch(fp_pairs)
    faith_scores_compact = nli_batch(faith_pairs)
    contra_scores_compact = _contradictions_batch(contra_pairs)

    faith_by_row: List[float] = [float("nan")] * len(base_rows)
    contra_by_row: List[float] = [float("nan")] * len(base_rows)
    for k, row_idx in enumerate(faith_indices):
        faith_by_row[row_idx] = faith_scores_compact[k]
        contra_by_row[row_idx] = contra_scores_compact[k]

    out: List[dict] = []
    for row, fp, faith, contra in zip(base_rows, fp_scores, faith_by_row, contra_by_row):
        abst_ok = 1.0 if (row["abstained"] == (row["fsr"] < 0.5)) else 0.0
        # Only non-abstained rows with a real faith score can register a
        # hallucination — abstained rows have NaN faith and are inert here.
        hallucination = (
            1.0
            if (not row["abstained"] and faith == faith and faith < 0.3)
            else 0.0
        )
        out.append(
            {
                "fact_span_recall": row["fsr"],
                "fact_span_precision": row["fsp"],
                "kpr": row["kpr"],
                "fact_precision": fp,
                "faithfulness": faith,
                "contradiction_rate": contra,
                "abstention_correct": abst_ok,
                "hallucination_flag": hallucination,
            }
        )
    return out


def _safe_mean(values: List[float]) -> float:
    """Mean that skips NaN entries (used by `_fact_span_precision` when there
    are no required chunks for a question)."""
    cleaned = [v for v in values if v == v]  # filters NaN
    return statistics.mean(cleaned) if cleaned else float("nan")


def _safe_median(values: List[float]) -> float:
    cleaned = [v for v in values if v == v]
    return statistics.median(cleaned) if cleaned else float("nan")


def run_evaluation(gt_path: str, retrieval_path: str, answers_path: str) -> int:
    print("\n" + "=" * 60)
    print("  RAG GT Pipeline V9.2 -- Phase 2: Evaluation")
    print("=" * 60)
    print(f"  GT file        : {gt_path}")
    print(f"  Retrieval logs : {retrieval_path}")
    print(f"  Answer logs    : {answers_path}")
    print("=" * 60 + "\n")

    import os

    corpus_name = os.path.splitext(os.path.basename(gt_path))[0]
    in_dir = os.path.dirname(gt_path) or None
    questions = load_gt(corpus_name, in_dir=in_dir)
    q_map = {q.q_id: q for q in questions}
    print(f"  Loaded {len(questions)} GT questions\n")

    def _load_jsonl(path: str) -> list[dict]:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    ret_raw = {
        d["q_id"]: RetrievalLog(
            q_id=d["q_id"], retrieved_chunk_ids=d["retrieved_chunk_ids"]
        )
        for d in _load_jsonl(retrieval_path)
    }
    ans_raw = {
        d["q_id"]: AnswerLog(
            q_id=d["q_id"],
            predicted_answer=d["predicted_answer"],
            abstained=d.get("abstained", False),
        )
        for d in _load_jsonl(answers_path)
    }

    matched = sum(1 for q in q_map if q in ret_raw and q in ans_raw)
    unmatched = len(q_map) - matched
    logger.info(f"Evaluation: {matched} matched, {unmatched} unmatched")

    if matched == 0:
        print("  No matching Q&A pairs found.")
        return 1

    # Use the batched path; it produces identical numbers and scales linearly
    # with corpus size.
    with tqdm(total=matched, desc="Evaluating", unit="q") as pbar:
        results = _evaluate_corpus_batched(q_map, ret_raw, ans_raw)
        pbar.update(matched)

    print("=" * 60)
    print("  METRIC RESULTS")
    print("=" * 60)
    bar_width = 30
    for key in METRIC_KEYS:
        values = [r[key] for r in results]
        mean_val = _safe_mean(values)
        median_val = _safe_median(values)
        # NaN means the metric was not applicable for any question.
        if mean_val != mean_val:
            print(f"  [N/A] {key:<25s} | (no applicable questions)")
            continue
        filled = int(max(0.0, min(1.0, mean_val)) * bar_width)
        bar = "#" * filled + "." * (bar_width - filled)
        if key in ("hallucination_flag", "contradiction_rate"):
            status = (
                "[OK]" if mean_val < 0.3 else "[WARN]" if mean_val < 0.6 else "[FAIL]"
            )
        else:
            status = (
                "[OK]" if mean_val >= 0.6 else "[WARN]" if mean_val >= 0.3 else "[FAIL]"
            )
        print(
            f"  {status} {key:<25s} |{bar}| mean={mean_val:.3f} median={median_val:.3f}"
        )
    print("=" * 60 + "\n")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        prog="rag-gt-evaluate",
        description="RAG GT Pipeline V9.2 -- Phase 2 Evaluation",
    )
    p.add_argument("--gt", required=True, help="Path to GT JSONL file")
    p.add_argument("--retrieval", required=True, help="Path to retrieval_logs.jsonl")
    p.add_argument("--answers", required=True, help="Path to answer_logs.jsonl")
    args = p.parse_args()
    sys.exit(run_evaluation(args.gt, args.retrieval, args.answers))


if __name__ == "__main__":
    main()
