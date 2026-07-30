"""Block: evaluator [FREE] -- qa + index + facts -> eval.

Wraps rag_gt.rag.eval_v2.v2_pair_to_eval + rag_gt.rag.matcher.match_pair +
rag_gt.rag.metrics.fill_metrics/aggregate verbatim (05_BLOCK_CATALOG.md
§3.28). The retriever is rebuilt from the index artifact's manifest (see
rag_gt.blocks.index_builder) by calling build_retriever again on the
referenced chunks.

Only match_mode="overlap" is supported: "exact-id" matching is called out in
the catalog as a not-yet-built Phase-1 item (rag_gt.rag.matcher has no
exact-id path today), so requesting it raises rather than silently
falling back to token-overlap matching.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

from rag_gt.blocks._common import artifact, read_json_artifact, read_list_input, write_json_artifact
from rag_gt.rag.eval_v2 import v2_pair_to_eval
from rag_gt.rag.matcher import match_pair
from rag_gt.rag.metrics import aggregate, fill_metrics
from rag_gt.rag.retriever import build_retriever


def run(inputs: dict, params: dict, artifacts_dir: Path | str | None = None) -> dict:
    match_mode = str(params.get("match_mode") or "overlap")
    if match_mode != "overlap":
        raise NotImplementedError(
            f"evaluator match_mode={match_mode!r} is not implemented in "
            "rag_gt.rag.matcher yet (catalog Phase-1 P1.4); use 'overlap'."
        )
    top_k = int(params.get("top_k", 10))

    facts = read_list_input(inputs.get("facts"))
    fact_text_by_id = {
        str(f.get("fact_id")): str(f.get("canonical_form") or f.get("text") or "")
        for f in facts
    }

    index_artifact = inputs.get("index")
    manifest = read_json_artifact(index_artifact["ref"])
    chunks = read_json_artifact(manifest["chunks_ref"])
    retriever = build_retriever(
        chunks, strategy=manifest["strategy"], embed_source=manifest.get("embed_source", "local")
    )
    id_to_text = {c.get("chunk_id", ""): c.get("text", "") for c in chunks}

    qa_pairs = read_list_input(inputs.get("qa"))
    rows = []
    for pair in qa_pairs:
        eval_pair = v2_pair_to_eval(pair, fact_text_by_id)
        ranked = retriever.retrieve(eval_pair["question"], top_k=top_k)
        rows.append(fill_metrics(match_pair(eval_pair, ranked, id_to_text)))

    agg = aggregate(rows)
    summary = agg.summary()
    by_pair_type = {pt: a.summary() for pt, a in sorted(agg.by_pair_type.items())}

    payload = {
        "top_k": top_k,
        "summary": summary,
        "by_pair_type": by_pair_type,
        "rows": [dataclasses.asdict(r) for r in rows],
    }
    ref = write_json_artifact(artifacts_dir, "evaluator", payload)
    return {"eval": artifact("eval", str(ref), {**summary, "top_k": top_k})}
