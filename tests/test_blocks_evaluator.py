"""TDD: rag_gt.blocks.evaluator wraps rag_gt.rag.eval_v2.v2_pair_to_eval +
rag_gt.rag.matcher.match_pair + rag_gt.rag.metrics.fill_metrics/aggregate
verbatim (05_BLOCK_CATALOG.md §3.28). qa + index + facts -> eval.

The index artifact only carries a manifest (see test_blocks_index_builder.py
docstring); evaluator rebuilds the retriever from the manifest's chunks_ref
via build_retriever -- same engine call index_builder itself made.
"""
from __future__ import annotations

import json

from rag_gt.rag.eval_v2 import v2_pair_to_eval
from rag_gt.rag.matcher import match_pair
from rag_gt.rag.metrics import aggregate, fill_metrics
from rag_gt.rag.retriever import build_retriever

CHUNKS = [
    {"chunk_id": "c1", "doc_id": "d", "text": "The sky is blue during a clear day.", "page_start": 1, "page_end": 1},
    {"chunk_id": "c2", "doc_id": "d", "text": "Grass is green in most seasons.", "page_start": 2, "page_end": 2},
]

FACTS = [
    {"fact_id": "F1", "text": "The sky is blue during a clear day.", "canonical_form": "The sky is blue during a clear day."},
    {"fact_id": "F2", "text": "Grass is green in most seasons.", "canonical_form": "Grass is green in most seasons."},
]

QA_PAIRS = [
    {
        "qa_id": "Q1", "question": "What color is the sky?", "evidence_strategy": "neighbor_pair_v2",
        "gold_fact_ids": ["F1"], "gold_pages": [1], "necessity": {"necessity_score": 1.0},
    },
    {
        "qa_id": "Q2", "question": "What color is grass in most seasons?", "evidence_strategy": "neighbor_pair_v2",
        "gold_fact_ids": ["F2"], "gold_pages": [2], "necessity": {"necessity_score": 1.0},
    },
]


def _inputs(tmp_path):
    from rag_gt.blocks._common import artifact, write_json_artifact
    from rag_gt.blocks.index_builder import run as index_builder_run

    chunks_ref = write_json_artifact(tmp_path, "chunks_import", CHUNKS)
    chunks_artifact = artifact("chunks", str(chunks_ref), {"count": len(CHUNKS), "doc_id": "d"})

    index_out = index_builder_run({"chunks": chunks_artifact}, {"strategy": "bm25"}, artifacts_dir=tmp_path)

    facts_ref = write_json_artifact(tmp_path, "facts_import", FACTS)
    facts_artifact = artifact("facts", str(facts_ref), {"count": len(FACTS), "grounded": True})

    qa_ref = write_json_artifact(tmp_path, "qa_fixture", QA_PAIRS)
    qa_artifact = artifact("qa", str(qa_ref), {"count": len(QA_PAIRS)})

    return {"qa": qa_artifact, "index": index_out["index"], "facts": facts_artifact}


def test_evaluator_returns_eval_artifact(tmp_path):
    from rag_gt.blocks.evaluator import run

    out = run(_inputs(tmp_path), {"top_k": 5}, artifacts_dir=tmp_path)

    assert set(out.keys()) == {"eval"}
    artifact = out["eval"]
    assert artifact["type"] == "eval"
    assert artifact["meta"]["n_pairs"] == 2
    assert artifact["meta"]["fact_recall@5"] == 1.0  # both facts are trivially retrievable


def test_evaluator_matches_manual_matcher_pipeline(tmp_path):
    from rag_gt.blocks.evaluator import run

    out = run(_inputs(tmp_path), {"top_k": 5}, artifacts_dir=tmp_path)

    # Reproduce the exact engine pipeline evaluator wraps, independently.
    lookup = {f["fact_id"]: f["canonical_form"] for f in FACTS}
    retriever = build_retriever(CHUNKS, strategy="bm25")
    id_to_text = {c["chunk_id"]: c["text"] for c in CHUNKS}
    results = []
    for pair in QA_PAIRS:
        eval_pair = v2_pair_to_eval(pair, lookup)
        ranked = retriever.retrieve(eval_pair["question"], top_k=5)
        results.append(fill_metrics(match_pair(eval_pair, ranked, id_to_text)))
    expected_summary = aggregate(results).summary()

    written = json.loads(open(out["eval"]["ref"], encoding="utf-8").read())
    assert written["summary"] == expected_summary
    assert out["eval"]["meta"] == {**expected_summary, "top_k": 5}
