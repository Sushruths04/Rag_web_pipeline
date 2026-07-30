"""Block adapter: **Neighbor QA Generator** ``qa_gen_pairs`` [PAID].

``candidates + facts -> qa``. Thin wrapper around the M4-refactored
``draft_neighbor_pairs`` (drafting) + ``gate_neighbor_pairs`` (NLI/necessity
gating) -- see ``05_BLOCK_CATALOG.md`` sec. 3 item 15 and sec. 4.

This block makes real LLM calls when actually invoked (one draft call per
candidate pair, minus cache hits). Never call ``run()`` in a test without
supplying ``params["llm"]``.
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
    draft_neighbor_pairs,
    gate_neighbor_pairs,
)
from rag_gt.validation.nli_check import nli_batch


def run(inputs: dict, params: dict, artifacts_dir: Path | str | None = None) -> dict:
    """``inputs``: ``{"facts": <facts artifact>, "candidates": <candidates artifact>}``.

    ``params``: ``doc`` (required), ``llm`` (optional -- FakeLLM in tests,
    else defaults to ``get_llm(params.get("llm_role", "gt"))``), ``workers``,
    ``cache_path`` (draft cache JSONL), ``require_chunk_ids`` (default True),
    ``nli_fn``/``necessity_fn``/``necessity_batch_fn`` (test overrides).

    Returns ``{"qa": {"type": "qa", "ref": <path>, "meta": {...}}}``.
    """
    facts = read_list_input(inputs["facts"])
    pairs = read_list_input(inputs["candidates"])
    doc = str(params["doc"])

    settings = _settings()
    doc_name = _doc_name(doc, settings)
    facts_by_id = {_fact_id(f): f for f in facts if _fact_id(f)}
    llm = params.get("llm") or get_llm(str(params.get("llm_role", "gt")))
    workers = int(params.get("workers") or settings["workers"])
    cache_path = Path(params["cache_path"]) if params.get("cache_path") else None

    drafts = draft_neighbor_pairs(
        pairs, facts_by_id, llm, settings,
        doc_name=doc_name, workers=workers, cache_path=cache_path,
    )

    thresholds = params.get("thresholds") or {
        "clause_min": float(settings["clause_entailment_min"]),
        "single_max": float(settings["single_fact_answer_max"]),
        "joint_min": float(settings["joint_answer_min"]),
    }
    nli_fn = params.get("nli_fn", nli_batch)
    necessity_fn = params.get("necessity_fn", leave_one_out_necessity)
    necessity_group_fn = _resolve_necessity_group_fn(necessity_fn, params.get("necessity_batch_fn"))

    accepted, rejected = gate_neighbor_pairs(
        pairs, facts_by_id, drafts, thresholds=thresholds,
        require_chunk_ids=bool(params.get("require_chunk_ids", True)),
        nli_fn=nli_fn, necessity_fn=necessity_group_fn,
        seen_questions=params.get("seen_questions", set()), doc=doc,
    )
    for index, item in enumerate(accepted, start=1):
        item["qa_id"] = f"V2Q{index:05d}"

    n_multihop = sum(1 for qa in accepted if qa.get("hop_type") != "single")
    meta = {"n_qa": len(accepted), "n_multihop": n_multihop,
            "rejected": dict(sorted(rejected.items()))}
    ref = write_json_artifact(artifacts_dir, "qa_gen_pairs", accepted)
    return {"qa": artifact("qa", str(ref), meta)}
