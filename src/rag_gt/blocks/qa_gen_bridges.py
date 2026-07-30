"""Block adapter: **Bridge QA Generator (2-fact)** ``qa_gen_bridges`` [PAID].

``bridges + facts -> qa``. Thin wrapper around
``build_answer_first_pairs(include_singles=False)`` -- see
``05_BLOCK_CATALOG.md`` sec. 3 item 17. Unlike ``qa_gen_pairs``/
``qa_gen_clusters``, this backing function was ALREADY a single
artifact-in/artifact-out callable before the M4 refactor (draft + gate +
assemble fused, but with a clean facts+bridge_pairs -> {"pairs","stats"}
boundary) -- see the M4 refactor task note: "don't force-split something
that doesn't have a clean boundary." No new draft_*/gate_* split was needed
for this stream.

This block makes real LLM calls when actually invoked (one draft call per
bridge pair, minus cache hits). Never call ``run()`` in a test without
supplying ``params["llm"]``.
"""
from __future__ import annotations

from pathlib import Path

from rag_gt.allpdf.necessity import leave_one_out_necessity
from rag_gt.blocks._common import artifact, read_list_input, write_json_artifact
from rag_gt.core.llm import get_llm
from rag_gt.generation.answer_first import build_answer_first_pairs
from rag_gt.validation.nli_check import nli_batch


def run(inputs: dict, params: dict, artifacts_dir: Path | str | None = None) -> dict:
    """``inputs``: ``{"facts": <facts artifact>, "bridges": <bridges artifact>}``.

    ``params``: ``llm`` (optional -- FakeLLM in tests, else defaults to
    ``get_llm(params.get("llm_role", "gt"))``), ``workers``, ``cache_path``
    (draft cache JSONL), ``require_chunk_ids`` (default False, matching
    ``build_answer_first_pairs``'s own default), ``clause_threshold``/
    ``single_max``/``joint_min`` (threshold overrides), ``nli_fn``/
    ``necessity_fn`` (test overrides -- single-item necessity contract, same
    as ``build_answer_first_pairs``).

    Returns ``{"qa": {"type": "qa", "ref": <path>, "meta": {...}}}``.
    """
    facts = read_list_input(inputs["facts"])
    bridge_pairs = read_list_input(inputs["bridges"])

    llm = params.get("llm") or get_llm(str(params.get("llm_role", "gt")))
    workers = params.get("workers")
    cache_path = Path(params["cache_path"]) if params.get("cache_path") else None
    nli_fn = params.get("nli_fn", nli_batch)
    necessity_fn = params.get("necessity_fn", leave_one_out_necessity)

    threshold_kwargs = {
        key: params[key]
        for key in ("clause_threshold", "single_max", "joint_min")
        if key in params
    }

    result = build_answer_first_pairs(
        facts, bridge_pairs, llm,
        include_singles=False, workers=workers, draft_cache_path=cache_path,
        require_chunk_ids=bool(params.get("require_chunk_ids", False)),
        nli_fn=nli_fn, necessity_fn=necessity_fn, **threshold_kwargs,
    )
    accepted = result["pairs"]

    meta = {"n_qa": len(accepted), "n_multihop": len(accepted), "stats": result["stats"]}
    ref = write_json_artifact(artifacts_dir, "qa_gen_bridges", accepted)
    return {"qa": artifact("qa", str(ref), meta)}
