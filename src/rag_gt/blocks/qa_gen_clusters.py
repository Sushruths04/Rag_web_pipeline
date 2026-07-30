"""Block adapter: **Cluster QA Generator (two-stage)** ``qa_gen_clusters`` [PAID].

``candidates + facts -> qa``. Thin wrapper around the M4-refactored
``draft_clusters`` (two-stage drafting: sides then question) +
``gate_clusters`` (NLI/necessity gating) -- see ``05_BLOCK_CATALOG.md``
sec. 3 item 16 and sec. 4.

This block makes real LLM calls when actually invoked (up to two draft
calls per side plus one question call per cluster, minus cache hits). Never
call ``run()`` in a test without supplying ``params["llm"]``.
"""
from __future__ import annotations

from pathlib import Path

from rag_gt.allpdf.necessity import leave_one_out_necessity
from rag_gt.blocks._common import artifact, read_list_input, write_json_artifact
from rag_gt.core.llm import get_llm
from rag_gt.generation.answer_first_v2 import (
    _doc_name,
    _fact_id,
    _resolve_necessity_group_fn,
    _settings,
    draft_clusters,
    gate_clusters,
)
from rag_gt.validation.nli_check import nli_batch


def run(inputs: dict, params: dict, artifacts_dir: Path | str | None = None) -> dict:
    """``inputs``: ``{"facts": <facts artifact>, "candidates": <candidates artifact>}``
    (candidates are 4-fact clusters from ``cluster_builder``).

    ``params``: ``doc`` (required), ``llm`` (optional -- FakeLLM in tests,
    else defaults to ``get_llm(params.get("llm_role", "gt"))``), ``workers``,
    ``cache_path`` (draft cache JSONL), ``require_chunk_ids`` (default True),
    ``nli_fn``/``necessity_fn``/``necessity_batch_fn`` (test overrides).

    Returns ``{"qa": {"type": "qa", "ref": <path>, "meta": {...}}}``.
    """
    facts = read_list_input(inputs["facts"])
    clusters = read_list_input(inputs["candidates"])
    doc = str(params["doc"])

    settings = _settings()
    doc_name = _doc_name(doc, settings)
    facts_by_id = {_fact_id(f): f for f in facts if _fact_id(f)}
    llm = params.get("llm") or get_llm(str(params.get("llm_role", "gt")))
    workers = int(params.get("workers") or settings["workers"])
    cache_path = Path(params["cache_path"]) if params.get("cache_path") else None

    thresholds = params.get("thresholds") or {
        "clause_min": float(settings["clause_entailment_min"]),
        "single_max": float(settings["single_fact_answer_max"]),
        "joint_min": float(settings["joint_answer_min"]),
    }
    nli_fn = params.get("nli_fn", nli_batch)

    drafts, draft_meta = draft_clusters(
        clusters, facts_by_id, llm, settings,
        workers=workers, cache_path=cache_path, nli_fn=nli_fn,
        doc_name=doc_name, clause_min=thresholds["clause_min"],
    )

    necessity_fn = params.get("necessity_fn", leave_one_out_necessity)
    necessity_group_fn = _resolve_necessity_group_fn(necessity_fn, params.get("necessity_batch_fn"))

    accepted, rejected = gate_clusters(
        clusters, facts_by_id, drafts, thresholds=thresholds,
        require_chunk_ids=bool(params.get("require_chunk_ids", True)),
        nli_fn=nli_fn, necessity_fn=necessity_group_fn,
        seen_questions=params.get("seen_questions", set()), doc=doc,
    )
    for index, item in enumerate(accepted, start=1):
        item["qa_id"] = f"V2Q{index:05d}"

    meta = {"n_qa": len(accepted), "n_multihop": len(accepted),
            "rejected": dict(sorted(rejected.items())), "draft_meta": draft_meta}
    ref = write_json_artifact(artifacts_dir, "qa_gen_clusters", accepted)
    return {"qa": artifact("qa", str(ref), meta)}
