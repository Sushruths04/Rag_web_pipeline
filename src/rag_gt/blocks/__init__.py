"""Block SDK adapters — thin ``run(inputs, params, artifacts_dir=None) -> dict``
wrappers around already-tested ``rag_gt`` engine functions.

Each module in this package wraps exactly one block from
``05_BLOCK_CATALOG.md`` §3. No generation internals are reimplemented here:
every module imports and calls the real engine function verbatim and only
handles artifact I/O (reading upstream JSON artifacts by ``ref`` path,
calling the engine function, writing the result to a new JSON artifact) plus
building the ``{"type": ..., "ref": ..., "meta": {...}}`` port-value
convention shared with ``studio/backend/stubs.py``.

Signature convention (kept identical across all block modules so the studio
adapter layer can wrap them uniformly):

    run(inputs: dict, params: dict, artifacts_dir: Path | str | None = None) -> dict

``inputs`` maps port name -> artifact dict (or ``None``/list, matching the
executor's ``resolve_inputs`` convention). ``params`` is a plain dict (block
parameters). ``artifacts_dir`` is the directory new output artifacts are
written under; if omitted, a process-wide temp directory is used so the
function is still safely callable on its own (e.g. from a script or test).

Coverage so far:
  - FREE spine (M0): chunks_import, facts_import, bridges_import, chunker,
    neighbor_sampler, cluster_builder, index_builder, evaluator, report.
  - PAID QA generation (M4): qa_gen_pairs, qa_gen_clusters, qa_gen_bridges --
    thin wrappers around the M4-refactored ``draft_*``/``gate_*`` steps in
    ``rag_gt.generation.answer_first_v2``. These make real LLM calls when
    actually invoked; never call ``run()`` in a test without a fake LLM in
    ``params["llm"]``.
"""
