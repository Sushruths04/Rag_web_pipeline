"""Quick analysis of v16 GT file — extracts ARM metrics, yield, QA-NLI stats.

Usage:
    python -m rag_gt.cli.analyze_v16_gt --gt data/gt/reinforcement_qa_source_v16_pass_gt_20260519.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from rag_gt.storage.gt_io import load_gt


def _doc_label(q) -> str:
    doc_ids = list(getattr(q, "doc_ids", []) or [])
    doc = doc_ids[0] if doc_ids else "unknown"
    return str(doc).split("/")[-1].replace(".pdf", "")


def main() -> None:
    p = argparse.ArgumentParser(prog="analyze-v16-gt")
    p.add_argument("--gt", required=True, help="v16 GT JSONL path")
    args = p.parse_args()

    gt_path = Path(args.gt)
    questions = load_gt(gt_path.stem, in_dir=gt_path.parent)

    # ---- Separate strict vs twins ----
    strict = [q for q in questions if q.twin_type is None]
    at_twins = [q for q in questions if q.twin_type == "abstention"]
    ct_twins = [q for q in questions if q.twin_type == "counterfactual"]

    print(f"\n{'='*60}")
    print(f"GT Yield Summary")
    print(f"{'='*60}")
    print(f"Total rows:   {len(questions)}")
    print(f"Strict:       {len(strict)}")
    print(f"AT twins:     {len(at_twins)}")
    print(f"CT twins:     {len(ct_twins)}")

    # ---- Per-corpus yield ----
    print(f"\n{'='*60}")
    print(f"Per-Corpus Yield")
    print(f"{'='*60}")
    corpus_strict: dict[str, int] = Counter()
    corpus_at: dict[str, int] = Counter()
    corpus_ct: dict[str, int] = Counter()
    corpus_distractors: dict[str, int] = Counter()

    for q in strict:
        doc = _doc_label(q)
        corpus_strict[doc] += 1
        corpus_distractors[doc] += len(q.distractor_spans or [])
    for q in at_twins:
        doc = _doc_label(q)
        corpus_at[doc] += 1
    for q in ct_twins:
        doc = _doc_label(q)
        corpus_ct[doc] += 1

    all_corpora = sorted(set(list(corpus_strict.keys()) + list(corpus_at.keys()) + list(corpus_ct.keys())))
    print(f"{'Corpus':<45} {'Strict':>7} {'AT':>5} {'CT':>5} {'Distract':>9}")
    print("-" * 75)
    for corp in all_corpora:
        print(f"{corp:<45} {corpus_strict[corp]:>7} {corpus_at[corp]:>5} {corpus_ct[corp]:>5} {corpus_distractors[corp]:>9}")
    print("-" * 75)
    print(f"{'TOTAL':<45} {len(strict):>7} {len(at_twins):>5} {len(ct_twins):>5} {sum(corpus_distractors.values()):>9}")

    # ---- QA-NLI gate statistics ----
    print(f"\n{'='*60}")
    print(f"QA-NLI Gate Statistics")
    print(f"{'='*60}")
    assumption_counts: dict[str, list] = defaultdict(list)
    total_assumptions = 0
    failed_assumptions = 0

    for q in strict:
        if q.question_assumptions:
            for assumption in q.question_assumptions:
                atype = assumption.get("type", "UNKNOWN")
                score = float(assumption.get("nli_score", 1.0))
                assumption_counts[atype].append(score)
                total_assumptions += 1
                if score < 0.55:
                    failed_assumptions += 1

    if total_assumptions > 0:
        print(f"Total assumptions checked: {total_assumptions}")
        print(f"Failed (<0.55 NLI):        {failed_assumptions} ({failed_assumptions/total_assumptions*100:.1f}%)")
        for atype, scores in sorted(assumption_counts.items()):
            fails = sum(1 for s in scores if s < 0.55)
            print(f"  {atype}: {len(scores)} checks, {fails} failed ({fails/len(scores)*100:.1f}%)")
    else:
        print("No question_assumptions found in strict rows.")

    # ---- ARM metrics ----
    print(f"\n{'='*60}")
    print(f"ARM (Abstractive Reasoning Map) Metrics")
    print(f"{'='*60}")
    np_overlaps: list[float] = []
    nli_scores: list[float] = []
    steps_passing: int = 0

    for q in strict:
        if q.reasoning_trace:
            for step in q.reasoning_trace:
                np_ov = step.get("np_overlap")
                nli_sc = step.get("nli_score")
                if np_ov is not None:
                    np_overlaps.append(float(np_ov))
                if nli_sc is not None:
                    nli_scores.append(float(nli_sc))
                    if float(nli_sc) >= 0.55:
                        steps_passing += 1

    if np_overlaps:
        mean_np = statistics.mean(np_overlaps)
        std_np = statistics.stdev(np_overlaps) if len(np_overlaps) > 1 else 0.0
        print(f"Total reasoning steps:   {len(np_overlaps)}")
        print(f"Mean NP overlap:         {mean_np:.3f}  (target ≤ 0.35) {'✓ PASS' if mean_np <= 0.35 else '✗ FAIL'}")
        print(f"Std NP overlap:          {std_np:.3f}")
        if nli_scores:
            mean_nli = statistics.mean(nli_scores)
            pct_pass = steps_passing / len(nli_scores) * 100
            print(f"Mean per-step NLI:       {mean_nli:.3f}  (target ≥ 0.55) {'✓ PASS' if mean_nli >= 0.55 else '✗ FAIL'}")
            print(f"% steps passing NLI:     {pct_pass:.1f}%")
    else:
        print("No reasoning_trace steps found. ARM may not be enabled.")

    # ---- Hop distribution ----
    print(f"\n{'='*60}")
    print(f"Chain Depth Distribution")
    print(f"{'='*60}")
    hop_counts: Counter = Counter()
    for q in strict:
        depth = len(q.required_fact_ids) if q.required_fact_ids else 1
        hop_counts[depth] += 1
    for depth in sorted(hop_counts.keys()):
        print(f"  hop={depth}: {hop_counts[depth]} rows ({hop_counts[depth]/len(strict)*100:.1f}%)")

    # ---- Intent distribution ----
    print(f"\n{'='*60}")
    print(f"Intent Distribution")
    print(f"{'='*60}")
    intent_counts: Counter = Counter()
    for q in strict:
        intent = getattr(q, "intent", None) or "unknown"
        intent_counts[intent] += 1
    for intent, count in intent_counts.most_common():
        print(f"  {intent}: {count} ({count/len(strict)*100:.1f}%)")

    # ---- TF-SFG edge statistics ----
    print(f"\n{'='*60}")
    print(f"TF-SFG Edge Statistics")
    print(f"{'='*60}")
    rows_with_edges = sum(1 for q in strict if q.tf_sfg_edges)
    total_edges = sum(len(q.tf_sfg_edges or []) for q in strict)
    edge_types: Counter = Counter()
    for q in strict:
        for edge in (q.tf_sfg_edges or []):
            edge_types[edge.get("type", "unknown")] += 1
    print(f"Rows with edges:   {rows_with_edges}/{len(strict)}")
    print(f"Total edges:       {total_edges}")
    if edge_types:
        print("Edge type distribution:")
        for etype, count in edge_types.most_common(5):
            print(f"  {etype}: {count}")

    # ---- Provenance / bbox coverage ----
    print(f"\n{'='*60}")
    print("Provenance / BBox Coverage")
    print(f"{'='*60}")
    total_spans = bbox_spans = page_spans = char_spans = source_spans = 0
    for q in questions:
        for fact in q.required_facts or []:
            for span in fact.supporting_spans or []:
                total_spans += 1
                if span.bboxes:
                    bbox_spans += 1
                if span.page_start is not None:
                    page_spans += 1
                if span.char_start is not None and span.char_end is not None:
                    char_spans += 1
                if span.source_path:
                    source_spans += 1
    if total_spans:
        print(f"Total spans:       {total_spans}")
        print(f"With bbox:         {bbox_spans} ({bbox_spans/total_spans*100:.1f}%)")
        print(f"With page:         {page_spans} ({page_spans/total_spans*100:.1f}%)")
        print(f"With char offsets: {char_spans} ({char_spans/total_spans*100:.1f}%)")
        print(f"With source path:  {source_spans} ({source_spans/total_spans*100:.1f}%)")
    else:
        print("No supporting spans found.")


if __name__ == "__main__":
    main()
