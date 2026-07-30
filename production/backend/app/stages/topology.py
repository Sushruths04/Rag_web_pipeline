"""The GRAFT pipeline shape. Single source of truth for stage names, labels,
dependencies, and tracks — used by the runner, the dummy pipeline, the real
stage registry (Plan 3/4), and mirrored by the frontend graph layout (Plan 2).
"""
from app.orchestrator.dag import StageDef

GRAFT_TOPOLOGY: list[StageDef] = [
    # Track 1 — ground truth. Preflight profiling comes FIRST: the profile
    # decides which ingestion backend and chunking strategy each PDF gets.
    StageDef("profile", "Structure Profile", (), "gt"),
    StageDef("ingest", "Ingest & Parse", ("profile",), "gt"),
    StageDef("chunk", "Adaptive Chunking", ("ingest",), "gt"),
    StageDef("extract", "Fact Extraction", ("chunk",), "gt"),
    StageDef("clean", "Fact Cleaning", ("extract",), "gt"),
    StageDef("graph", "Bridge Graph Build", ("clean",), "gt"),
    StageDef("sample", "Multi-hop Sampling", ("graph",), "gt"),
    StageDef("qagen", "QA Generation (LLM)", ("sample",), "gt"),
    StageDef("verify", "Verification", ("qagen",), "gt"),
    StageDef("gt_dataset", "GT Dataset", ("verify",), "gt"),
    # Track 2 — RAG build (branches off chunking)
    StageDef("bm25_index", "BM25 Index", ("chunk",), "rag"),
    StageDef("vector_index", "Vector Index", ("chunk",), "rag"),
    StageDef("fusion", "Hybrid Fusion", ("bm25_index", "vector_index"), "rag"),
    StageDef("rerank_warmup", "Reranker Warm-up", ("fusion",), "rag"),
    # Track 3 — evaluation & self-improvement
    StageDef("sweep", "Auto-tune Sweep", ("gt_dataset", "fusion", "rerank_warmup"), "eval"),
    StageDef("leaderboard", "Leaderboard", ("sweep",), "eval"),
    StageDef("select_config", "Best-Config Selection", ("leaderboard",), "eval"),
    StageDef("final_eval", "Final Eval Run", ("select_config",), "eval"),
    StageDef("report", "Report", ("final_eval",), "eval"),
]
