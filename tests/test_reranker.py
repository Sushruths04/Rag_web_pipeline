"""Reranking tests: standard relevance vs inference-safe complementarity."""

from rag_gt.rag.reranker import CrossEncoderReranker, CrossEncoderScorer


class BaseRetriever:
    def __init__(self):
        self.chunks = {
            "c1": {"chunk_id": "c1", "text": "alpha pressure requirement", "page_start": 1},
            "c2": {"chunk_id": "c2", "text": "alpha pressure requirement restated", "page_start": 1},
            "c3": {"chunk_id": "c3", "text": "beta shutdown consequence", "page_start": 2},
        }

    def retrieve(self, _query, top_k=10):
        return [("c1", 1.0), ("c2", 0.9), ("c3", 0.8)][:top_k]

    def get_chunk(self, chunk_id):
        return self.chunks[chunk_id]

    def get_chunk_text(self, chunk_id):
        return self.chunks[chunk_id]["text"]


class FakeModel:
    def predict(self, pairs, **_kwargs):
        return [
            2.9 if "restated" in text else 2.8 if "beta" in text else 3.0
            for _, text in pairs
        ]


def test_standard_reranker_keeps_top_cross_encoder_scores():
    reranker = CrossEncoderReranker(
        BaseRetriever(), CrossEncoderScorer(FakeModel()), mode="relevance"
    )
    assert [item[0] for item in reranker.retrieve("alpha and beta", top_k=2)] == [
        "c1",
        "c2",
    ]


def test_bridge_reranker_prefers_complementary_cross_page_evidence():
    reranker = CrossEncoderReranker(
        BaseRetriever(),
        CrossEncoderScorer(FakeModel()),
        mode="bridge_complementarity",
    )
    assert [item[0] for item in reranker.retrieve("alpha and beta", top_k=2)] == [
        "c1",
        "c3",
    ]
