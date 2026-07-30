from app.evaluation.matching import (
    aggregate,
    pick_winner,
    precision_rw,
    score_pair,
    tokenize,
    unit_hit,
)


def test_tokenize_lowercase_alnum():
    assert tokenize("The ISO 15607 spec (v2)!") == ["the", "iso", "15607", "spec", "v2"]


def test_unit_hit_full_containment():
    unit = "hardness is measured on the Vickers scale"
    chunk = "In this method, hardness is measured on the Vickers scale using a diamond indenter."
    assert unit_hit(unit, [chunk]) is True


def test_unit_hit_requires_60_percent_in_single_chunk():
    unit = "alpha beta gamma delta epsilon"  # 5 tokens; needs >= 3 in ONE chunk
    assert unit_hit(unit, ["alpha beta something"]) is False        # 2/5
    assert unit_hit(unit, ["alpha beta gamma other"]) is True       # 3/5
    # split across chunks does NOT count
    assert unit_hit(unit, ["alpha beta", "gamma delta"]) is False


def _pair(clauses, answer="fallback answer text"):
    return {
        "qa_id": "q1",
        "hop_type": "single",
        "answer": answer,
        "answer_clauses": [{"text": t} for t in clauses],
    }


def test_score_pair_recall_precision():
    pair = _pair(["alpha beta gamma delta", "omega psi chi phi"])
    retrieved = [
        "alpha beta gamma delta and more context",  # covers unit 1
        "totally unrelated chunk about nothing",     # irrelevant
    ]
    row = score_pair(pair, retrieved)
    assert row["recall"] == 0.5          # 1 of 2 units
    assert row["precision"] == 0.5       # 1 of 2 chunks relevant
    assert row["hit"] is False
    assert row["n_units"] == 2


def test_score_pair_full_hit():
    pair = _pair(["alpha beta gamma delta"])
    row = score_pair(pair, ["alpha beta gamma delta exactly here"])
    assert row["recall"] == 1.0 and row["precision"] == 1.0 and row["hit"] is True


def test_score_pair_falls_back_to_answer():
    pair = {"qa_id": "q2", "hop_type": "bridge", "answer": "omega psi chi phi", "answer_clauses": []}
    row = score_pair(pair, ["omega psi chi phi contained right here"])
    assert row["recall"] == 1.0 and row["n_units"] == 1


def test_aggregate_means_and_hop_breakdown():
    rows = [
        {"qa_id": "a", "hop_type": "single", "recall": 1.0, "precision": 0.5, "precision_rw": 1.0, "hit": True, "n_units": 1},
        {"qa_id": "b", "hop_type": "single", "recall": 0.0, "precision": 0.0, "precision_rw": 0.0, "hit": False, "n_units": 1},
        {"qa_id": "c", "hop_type": "bridge", "recall": 1.0, "precision": 1.0, "precision_rw": 1.0, "hit": True, "n_units": 2},
    ]
    agg = aggregate(rows)
    assert agg["n"] == 3
    assert abs(agg["recall"] - 2 / 3) < 1e-9
    assert abs(agg["precision"] - 0.5) < 1e-9
    assert abs(agg["hit_rate"] - 2 / 3) < 1e-9
    expected_f1 = 2 * (2 / 3) * 0.5 / (2 / 3 + 0.5)
    assert abs(agg["f1"] - expected_f1) < 1e-9
    assert agg["by_hop"]["single"]["n"] == 2
    assert agg["by_hop"]["bridge"]["recall"] == 1.0


def test_aggregate_empty():
    agg = aggregate([])
    assert agg["n"] == 0 and agg["f1"] == 0.0


def test_precision_rw_relevant_at_rank_one_is_perfect():
    assert precision_rw([1, 0, 0, 0, 0]) == 1.0


def test_precision_rw_relevant_at_rank_k():
    assert abs(precision_rw([0, 0, 1]) - 1 / 3) < 1e-9


def test_precision_rw_multiple_relevant():
    # ranks 1 and 3 relevant: (1/1 + 2/3) / 2
    assert abs(precision_rw([1, 0, 1]) - (1.0 + 2 / 3) / 2) < 1e-9


def test_precision_rw_none_relevant_or_empty():
    assert precision_rw([0, 0, 0]) == 0.0
    assert precision_rw([]) == 0.0


def test_score_pair_single_evidence_not_punished_at_k5():
    # THE bug this phase fixes: 1 gold unit, retrieved at rank 1 of 5.
    pair = _pair(["alpha beta gamma delta"])
    retrieved = ["alpha beta gamma delta here", "noise one", "noise two", "noise three", "noise four"]
    row = score_pair(pair, retrieved)
    assert row["precision"] == 0.2          # raw metric still reported
    assert row["precision_rw"] == 1.0       # fair metric: evidence ranked first
    assert row["rel_at_k"] == [1, 0, 0, 0, 0]


def test_aggregate_has_rw_metrics_and_by_hop():
    rows = [
        {"qa_id": "a", "hop_type": "single", "recall": 1.0, "precision": 0.2,
         "precision_rw": 1.0, "hit": True, "n_units": 1},
        {"qa_id": "b", "hop_type": "bridge", "recall": 0.5, "precision": 0.4,
         "precision_rw": 0.5, "hit": False, "n_units": 2},
    ]
    agg = aggregate(rows)
    assert abs(agg["precision_rw"] - 0.75) < 1e-9
    expected_f1_rw = 2 * 0.75 * 0.75 / (0.75 + 0.75)
    assert abs(agg["f1_rw"] - expected_f1_rw) < 1e-9
    assert agg["by_hop"]["single"]["precision_rw"] == 1.0
    assert agg["by_hop"]["bridge"]["precision_rw"] == 0.5


def test_pick_winner_by_f1_rw_then_smaller_k():
    rows = [
        {"config": "a@3", "top_k": 3, "f1": 0.9, "f1_rw": 0.5},
        {"config": "b@5", "top_k": 5, "f1": 0.4, "f1_rw": 0.8},
        {"config": "c@10", "top_k": 10, "f1": 0.4, "f1_rw": 0.8},
    ]
    assert pick_winner(rows)["config"] == "b@5"  # f1_rw wins; k=5 beats k=10 on tie
