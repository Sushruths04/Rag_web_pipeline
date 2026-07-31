"""Decomposition retriever: per-sub-question retrieval + RRF, inference-safe ($0)."""

from rag_gt.rag.decomposition import DecompositionRetriever


class FakeBase:
    def __init__(self):
        self.chunks = {
            "cA": {"chunk_id": "cA", "text": "alpha pressure on page 1"},
            "cB": {"chunk_id": "cB", "text": "beta shutdown on page 2"},
            "cN": {"chunk_id": "cN", "text": "unrelated noise"},
        }

    def retrieve(self, query, top_k=10):
        q = query.lower()
        # The un-decomposed bridge query (mentions both) retrieves poorly — the
        # motivation for decomposition. Each single-hop sub-query retrieves its hop.
        if "alpha" in q and "beta" in q:
            return []
        if "alpha" in q:
            return [("cA", 1.0)][:top_k]
        if "beta" in q:
            return [("cB", 1.0)][:top_k]
        return []

    def get_chunk(self, cid):
        return self.chunks[cid]

    def get_chunk_text(self, cid):
        return self.chunks[cid]["text"]


class SplittingLLM:
    def generate_json(self, prompt, temperature=0.0, max_tokens=0):
        return {"sub_questions": ["what about alpha pressure?", "what about beta shutdown?"]}


class NoSplitLLM:
    def generate_json(self, prompt, temperature=0.0, max_tokens=0):
        return {"sub_questions": []}


def test_decomposition_surfaces_both_hops_via_rrf():
    r = DecompositionRetriever(FakeBase(), SplittingLLM())
    ids = [cid for cid, _ in r.retrieve("alpha and beta combined?", top_k=2)]
    assert "cA" in ids and "cB" in ids  # both hops recovered, not just noise


def test_falls_back_to_full_query_when_llm_returns_nothing():
    r = DecompositionRetriever(FakeBase(), NoSplitLLM())
    ids = [cid for cid, _ in r.retrieve("alpha only?", top_k=2)]
    assert ids[0] == "cA"  # full query still works


def test_sub_questions_are_cached():
    r = DecompositionRetriever(FakeBase(), SplittingLLM())
    r.retrieve("alpha and beta?", top_k=2)
    assert "alpha and beta?" in r._subq_cache
