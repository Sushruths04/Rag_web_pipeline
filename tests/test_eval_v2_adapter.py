"""v2 GT -> no-API eval adapter.

The matcher does token-overlap between FACT text and chunk text, so the
adapter must join the TRUE source-fact texts by exact gold_fact_ids — never
the answer clauses (paraphrases would degrade overlap and understate recall).
"""
import pytest

from rag_gt.rag.eval_v2 import doc_to_pipeline_id, v2_pair_to_eval

V2_PAIR = {
    "qa_id": "V2Q00001",
    "hop_type": "bridge",
    "evidence_strategy": "cluster_2plus2",
    "question": "Which values are defined, and which records name the edition?",
    "answer_clauses": [
        {"text": "paraphrased clause a", "fact_id": "d_F1", "nli": 0.9},
        {"text": "paraphrased clause b", "fact_id": "d_F2", "nli": 0.9},
    ],
    "gold_fact_ids": ["d_F1", "d_F2"],
    "gold_pages": [1, 3],
    "necessity": {"necessity_score": 1.0},
}

FACT_TEXTS = {
    "d_F1": "The welding procedure specification defines ranges for variables.",
    "d_F2": "Qualification records must reference the governing standard edition.",
}


def test_v2_pair_to_eval_uses_true_fact_texts():
    out = v2_pair_to_eval(V2_PAIR, FACT_TEXTS)
    assert out["question"] == V2_PAIR["question"]
    assert out["pair_type"] == "cluster_2plus2"
    assert out["depth"] == 2
    assert out["page_spread"] == 2
    assert out["necessity_score"] == 1.0
    texts = [f["text"] for f in out["facts"]]
    assert texts == [FACT_TEXTS["d_F1"], FACT_TEXTS["d_F2"]]
    assert "paraphrased" not in " ".join(texts)
    assert [f["fact_id"] for f in out["facts"]] == ["d_F1", "d_F2"]


def test_v2_pair_to_eval_unknown_fact_id_raises():
    with pytest.raises(KeyError):
        v2_pair_to_eval(V2_PAIR, {"d_F1": "only one"})


def test_doc_to_pipeline_id_strips_full_suffix():
    assert doc_to_pipeline_id("din_iso_6507_vickers_full") == "din_iso_6507_vickers"
    assert doc_to_pipeline_id("sutton_rl") == "sutton_rl"
