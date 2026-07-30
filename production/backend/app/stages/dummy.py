"""Simulated pipeline: real topology, fake work. Drives frontend development
and orchestrator tests without touching PDFs or the LLM API.

Config knobs: sleep_scale (float, default 0.05), fail_stage (str | None).
"""
from __future__ import annotations

import random
import time

from app.orchestrator.context import StageContext
from app.orchestrator.dag import PipelineDAG, StageDef

_UNITS = {  # stage -> (n work units, unit label)
    "ingest": (4, "pdf"), "profile": (4, "pdf"), "chunk": (8, "section"),
    "extract": (10, "page batch"), "clean": (6, "filter rule"),
    "graph": (5, "entity cluster"), "sample": (4, "chain batch"),
    "qagen": (5, "question"), "verify": (5, "pair"), "gt_dataset": (1, "write"),
    "bm25_index": (3, "shard"), "vector_index": (4, "batch"),
    "fusion": (2, "config"), "rerank_warmup": (2, "model"),
    "sweep": (8, "config cell"), "leaderboard": (1, "rank"),
    "select_config": (1, "pick"), "final_eval": (5, "question"),
    "report": (2, "section"),
}


def _simulate(ctx: StageContext) -> None:
    stage = ctx.stage
    sleep_scale = float(ctx.config.get("sleep_scale", 0.05))
    n, unit = _UNITS[stage]
    rng = random.Random(f"{ctx.run_id}/{stage}")

    ctx.log("info", f"{stage}: starting ({n} {unit}s)")
    for i in range(1, n + 1):
        ctx.check_cancel()
        time.sleep(sleep_scale * rng.uniform(0.5, 1.5))
        with ctx.span("processing", f"{unit}_{i}", input={"unit": unit, "i": i}) as sp:
            if stage == "qagen":
                with sp.child("llm", "generate_pair", input={"prompt": f"draft q{i}"}) as ch:
                    ch.output = {"text": f"answer {i}"}
                    ch.tokens_in = rng.randint(400, 900)
                    ch.tokens_out = rng.randint(80, 300)
                    ch.cost_usd = round(ch.tokens_in * 2e-7 + ch.tokens_out * 6e-7, 8)
            if stage == "vector_index":
                with sp.child("embedding", "embed_batch", input={"batch": i}) as ch:
                    ch.output = {"vectors": 64}
                    ch.extra = {"model": "all-MiniLM-L6-v2", "device": "cpu"}
            sp.output = {"ok": True}
        ctx.progress(i, n, f"{unit} {i}/{n}")

    if ctx.config.get("fail_stage") == stage:
        raise RuntimeError(f"simulated failure in {stage}")

    if stage == "extract":
        ctx.metric("facts_extracted", rng.randint(300, 700))
    if stage == "verify":
        ctx.metric("pairs_kept", rng.randint(40, 90))
        ctx.metric("llm_cost_usd", round(rng.uniform(0.5, 2.0), 4))
    if stage == "gt_dataset":
        p = ctx.run_dir / "artifacts" / "gt.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"question": "dummy?", "answer": "dummy."}\n', encoding="utf-8")
        ctx.artifact("gt.jsonl", p, kind="jsonl")
    ctx.log("info", f"{stage}: done")


def build_dummy_dag(topology: list[StageDef]) -> PipelineDAG:
    return PipelineDAG(
        [
            StageDef(s.name, s.label, s.deps, s.track, _simulate)
            for s in topology
        ]
    )
