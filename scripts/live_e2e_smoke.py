"""Live-mode (llm_mode="live") end-to-end smoke run of the "graft" pipeline.

Verifies TODO.md 3.4: actually exercise the real RunManager -> real GRAFT
stage wrappers -> real RWTH HPC LLM endpoint on this exported repo, using the
small ecma404 JSON-spec PDF, and report genuine per-stage status, real
token/cost figures recorded by TracedLLM/CostBudget, and the final QA output
count / eval metrics (or the exact failure).

Run from the worktree root:
    python scripts/live_e2e_smoke.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

WORKTREE_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = WORKTREE_ROOT / "production" / "backend"
PDF = WORKTREE_ROOT / "data" / "test_corpus_allpdf" / "ecma404_json.pdf"

sys.path.insert(0, str(BACKEND_ROOT))

RUN_DIR = WORKTREE_ROOT / "production" / "data" / "runs" / "_live_e2e_smoke"


def _fresh_run_dir() -> Path:
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    return RUN_DIR


def main() -> int:
    print(f"[smoke] worktree root: {WORKTREE_ROOT}")
    print(f"[smoke] PDF: {PDF} (exists={PDF.exists()})")
    if not PDF.exists():
        print("[smoke] FATAL: test PDF missing, aborting.")
        return 2

    from app.db.store import Store
    from app.orchestrator.events import EventBus
    from app.orchestrator.manager import RunManager

    run_root = _fresh_run_dir()
    store = Store(run_root / "db.sqlite")
    mgr = RunManager(store, EventBus(), run_root / "runs", mode="thread")

    config = {
        "docling_page_cap": 4,
        "llm_mode": "live",
        "price_in_per_mtok": 0.0,
        "price_out_per_mtok": 0.0,
        "max_cost_usd": 1.0,
        # keep the first live smoke run small/cheap on every tunable the
        # graft stages read (see production/backend/app/stages/graft.py)
        "llm_chunk_cap": 20,       # extract_stage: max chunks sent to the LLM
        "max_pairs": 60,           # graph_stage: candidate edge budget cap
        "target_chains": 8,        # sample_stage: single-fact chains sampled
        "multihop_chains": 4,      # sample_stage: extra typed multi-hop walks
        "score_necessity": False,  # qagen_stage: skip the extra LOO LLM pass
        "sweep_modes": ["bm25", "vector", "hybrid"],
        "sweep_top_ks": [3, 5],
        "sweep_sample": 40,
        "eval_span_cap": 25,
    }
    print(f"[smoke] config: {json.dumps(config, indent=2)}")

    run_id = mgr.prepare_run(config, pipeline="graft")
    uploads_dir = mgr.data_dir / run_id / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(PDF, uploads_dir / PDF.name)
    print(f"[smoke] run_id={run_id}, uploaded {PDF.name} -> {uploads_dir}")

    # Sanity check: confirm rag_gt resolves to THIS worktree's src, not the
    # globally pip-installed editable D:\Mini Thesis\RAG_GT copy (see
    # feedback_worktree_python_resolution memory note).
    sys.path.insert(0, str(WORKTREE_ROOT / "src"))
    import rag_gt  # noqa: E402

    print(f"[smoke] rag_gt resolved from: {rag_gt.__file__}")
    expected = str((WORKTREE_ROOT / "src" / "rag_gt" / "__init__.py").resolve())
    actual = str(Path(rag_gt.__file__).resolve())
    if actual != expected:
        print(
            f"[smoke] WARNING: rag_gt did not resolve to the worktree src!\n"
            f"  expected: {expected}\n  actual:   {actual}"
        )
    else:
        print("[smoke] OK: rag_gt resolves to the worktree's own src/ (not the main-repo editable install)")

    t0 = time.time()
    mgr.launch_run(run_id)

    # Poll so we can print stage transitions as they happen, rather than
    # only seeing the final state after a long silent wait.
    seen_status: dict[str, str] = {}
    timeout_s = 600.0
    deadline = t0 + timeout_s
    while True:
        run = store.get_run(run_id)
        for s in run["stages"]:
            name, status = s["name"], s["status"]
            if seen_status.get(name) != status:
                seen_status[name] = status
                elapsed = time.time() - t0
                extra = ""
                if status == "failed":
                    extra = f" ERROR: {s.get('error')}"
                print(f"[smoke] t={elapsed:6.1f}s  stage={name:16s} -> {status}{extra}")
        if run["status"] in ("completed", "failed", "cancelled"):
            break
        if time.time() > deadline:
            print("[smoke] TIMEOUT waiting for run to finish")
            break
        time.sleep(1.0)

    mgr.wait(run_id, timeout=5.0)
    elapsed_total = time.time() - t0
    run = store.get_run(run_id)

    print("\n" + "=" * 70)
    print(f"[smoke] FINAL run status: {run['status']}  (elapsed {elapsed_total:.1f}s)")
    print("=" * 70)

    stages = {s["name"]: s for s in run["stages"]}
    print("\n[smoke] Per-stage status:")
    for s in run["stages"]:
        err = f"  ERROR: {s.get('error')}" if s.get("error") else ""
        print(f"  {s['name']:16s} track={s.get('track', '?'):6s} status={s['status']:10s}{err}")
        if s.get("traceback"):
            print("  --- traceback ---")
            print("  " + s["traceback"].replace("\n", "\n  "))
            print("  -----------------")

    metrics = {m["name"]: m["value"] for m in store.list_metrics(run_id)}
    print("\n[smoke] Metrics recorded:")
    for k, v in sorted(metrics.items()):
        print(f"  {k} = {v}")

    spans = store.list_spans(run_id, limit=5000)
    llm_spans = [sp for sp in spans if sp.get("type") == "llm"]
    print(f"\n[smoke] LLM spans recorded: {len(llm_spans)}")
    total_tok_in = sum(sp.get("tokens_in") or 0 for sp in llm_spans)
    total_tok_out = sum(sp.get("tokens_out") or 0 for sp in llm_spans)
    total_cost = sum(sp.get("cost_usd") or 0.0 for sp in llm_spans)
    print(f"  total tokens_in  = {total_tok_in}")
    print(f"  total tokens_out = {total_tok_out}")
    print(f"  total cost_usd   = {total_cost}")
    if llm_spans:
        print("  sample span[0]:")
        sample = {k: v for k, v in llm_spans[0].items() if k not in ("input", "output")}
        print(f"    {json.dumps(sample, default=str, indent=4)}")
        print(f"    input.prompt (truncated): {str(llm_spans[0].get('input', {}).get('prompt', ''))[:200]!r}")
        print(f"    output.text (truncated):  {str(llm_spans[0].get('output', {}).get('text', ''))[:200]!r}")

    artifacts = {a["name"]: a for a in store.list_artifacts(run_id)}
    print(f"\n[smoke] Artifacts: {sorted(artifacts)}")

    gt_pairs = None
    if "gt.jsonl" in artifacts:
        gt_path = Path(artifacts["gt.jsonl"]["path"])
        lines = gt_path.read_text(encoding="utf-8").strip().splitlines()
        gt_pairs = [json.loads(l) for l in lines if l.strip()]
        print(f"\n[smoke] gt.jsonl: {len(gt_pairs)} QA pairs")
        if gt_pairs:
            print(f"  sample pair[0]: {json.dumps(gt_pairs[0], default=str)[:500]}")

    if "eval_results.json" in artifacts:
        ev = json.loads(Path(artifacts["eval_results.json"]["path"]).read_text(encoding="utf-8"))
        print(f"\n[smoke] eval_results.json aggregate: {json.dumps(ev.get('aggregate'), indent=2)}")
        print(f"  winner config: {ev.get('winner', {}).get('config')}")

    if "leaderboard.json" in artifacts:
        lb = json.loads(Path(artifacts["leaderboard.json"]["path"]).read_text(encoding="utf-8"))
        print(f"\n[smoke] leaderboard rows: {len(lb.get('rows', []))}")
        for row in lb.get("rows", []):
            marker = "*" if row.get("winner") else " "
            print(
                f"  {marker} {row['config']:18s} recall={row['recall']:.3f} "
                f"precision={row['precision']:.3f} f1={row['f1']:.3f} n={row['n']}"
            )

    print("\n[smoke] DONE")
    return 0 if run["status"] == "completed" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("[smoke] UNCAUGHT EXCEPTION in smoke script itself:")
        traceback.print_exc()
        sys.exit(3)
