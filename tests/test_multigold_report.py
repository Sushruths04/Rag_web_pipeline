"""Stage E multi-gold metrics, configured dense retrieval, and HTML tests."""

import json

import numpy as np

from rag_gt.rag.matcher import match_pair
from rag_gt.rag.metrics import fill_metrics
from rag_gt.rag.multigold_report import (
    enrich_qa_grounding,
    evaluate_multigold,
    render_html,
)
from rag_gt.rag.retriever import DenseRetriever


class FakeEmbedder:
    def encode(self, texts, **_kwargs):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "alpha" in lowered else 0.0,
                    1.0 if "beta" in lowered else 0.0,
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


def test_dense_retriever_uses_injected_configured_embedder():
    chunks = [
        {"chunk_id": "a", "text": "alpha rule"},
        {"chunk_id": "b", "text": "beta condition"},
    ]
    retriever = DenseRetriever(chunks, embedder=FakeEmbedder())
    retriever.prime_queries(["beta"])
    assert retriever.retrieve("beta", top_k=1)[0][0] == "b"


def test_precision_counts_relevant_chunks_not_gold_facts_twice():
    pair = {
        "question": "q",
        "pair_type": "bridge",
        "facts": [
            {"fact_id": "F1", "text": "alpha shared statement"},
            {"fact_id": "F2", "text": "beta separate condition"},
        ],
    }
    ranked = [("both", 1.0), ("noise", 0.5)]
    texts = {
        "both": "alpha shared statement and beta separate condition",
        "noise": "unrelated material",
    }
    result = fill_metrics(match_pair(pair, ranked, texts), k_values=[1, 2])
    assert result.fact_recall_at_k[1] == 1.0
    assert result.precision_at_k[1] == 1.0
    assert result.precision_at_k[2] == 0.5
    assert result.hit_at_k[1] == 1


def test_bm25_multigold_report_and_html(tmp_path):
    doc = "synthetic_full"
    checkpoint = tmp_path / doc / "checkpoints"
    checkpoint.mkdir(parents=True)
    chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "synthetic",
            "text": "Alpha pressure is the controlling parameter.",
            "page_start": 1,
            "page_end": 1,
            "source_path": "data/synthetic.pdf",
            "source_char_start": 0,
            "source_units": [
                {
                    "char_start": 0,
                    "char_end": 43,
                    "page_no": 1,
                    "bboxes": [
                        {
                            "page_no": 1,
                            "l": 10,
                            "t": 20,
                            "r": 200,
                            "b": 40,
                            "coord_origin": "TOPLEFT",
                        }
                    ],
                }
            ],
        },
        {
            "chunk_id": "c2",
            "doc_id": "synthetic",
            "text": "Exceeding the beta limit requires shutdown.",
            "page_start": 2,
            "page_end": 2,
            "source_path": "data/synthetic.pdf",
            "source_char_start": 0,
            "source_units": [
                {
                    "char_start": 0,
                    "char_end": 43,
                    "page_no": 2,
                    "bboxes": [
                        {
                            "page_no": 2,
                            "l": 10,
                            "t": 50,
                            "r": 220,
                            "b": 70,
                            "coord_origin": "TOPLEFT",
                        }
                    ],
                }
            ],
        },
        {
            "chunk_id": "c3",
            "doc_id": "synthetic",
            "text": "Unrelated maintenance schedule.",
            "page_start": 3,
            "page_end": 3,
        },
    ]
    (checkpoint / "s2_chunks_full.json").write_text(json.dumps(chunks), encoding="utf-8")
    facts = [
        {"id": "F1", "text": chunks[0]["text"], "page": 1, "doc": doc},
        {"id": "F2", "text": chunks[1]["text"], "page": 2, "doc": doc},
    ]
    qa = {
        "qa_id": "Q1",
        "doc": doc,
        "hop_type": "bridge",
        "bridge_entity": "control system",
        "question": "Which pressure parameter and limit determine when shutdown is required?",
        "answer": "Alpha pressure controls the process; exceeding beta requires shutdown.",
        "gold_fact_ids": ["F1", "F2"],
        "gold_pages": [1, 2],
        "gold_bboxes": {"F1": [], "F2": []},
        "necessity": {"necessity_score": 1.0},
        "verify": {"verdict": "PASS"},
    }
    report = evaluate_multigold(facts, [qa], tmp_path, strategies=["bm25"])
    assert report["metadata"]["n_evaluated"] == 1
    assert report["summary"]["bm25"]["bridge"]["n_pairs"] == 1
    assert "precision@5" in report["summary"]["bm25"]["bridge"]
    assert "hop_completeness@5" in report["summary"]["bm25"]["bridge"]
    assert len(report["per_qa"][0]["gold_supports"]) == 2
    assert report["metadata"]["bbox_coverage"] == 1.0
    enriched = enrich_qa_grounding({"pairs": [qa], "stats": {}}, report)
    assert enriched["pairs"][0]["gold_chunk_ids"] == ["c1", "c2"]
    assert enriched["pairs"][0]["gold_bboxes"]["F1"][0]["page_no"] == 1
    rendered = render_html(report)
    assert "Multi-gold Retrieval Report" in rendered
    assert "Precision@5" in rendered
    assert "bbox (approximate_char_overlap, alignment" in rendered
