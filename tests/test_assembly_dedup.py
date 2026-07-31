"""Near-dupe + shared-evidence dedup at final assembly (T4).

Two independent dedup passes, both scoped to the same document:
(a) exact shared-evidence — pairs whose sorted gold_fact_ids sets are
    identical collapse to a single survivor (more evidence units wins, then
    higher min clause NLI);
(b) near-dupe question — pairs whose question embeddings have cosine
    similarity >= 0.92 (same doc) collapse via the same keep-rule.

The embedder is injected (embed_fn) so these tests never load a real
sentence-transformer.
"""
from rag_gt.generation.dataset_budget import dedup_pairs


def _qa(qa_id, doc, question, fact_ids, nli_scores):
    return {
        "qa_id": qa_id,
        "doc": doc,
        "question": question,
        "gold_fact_ids": fact_ids,
        "answer_clauses": [
            {"text": f"clause {i}", "fact_id": fid, "nli": score}
            for i, (fid, score) in enumerate(zip(fact_ids, nli_scores))
        ],
    }


class _FakeEmbedder:
    """Maps known question strings to hand-picked vectors: the two paraphrases
    point in nearly the same direction (cosine ~0.999), the control points
    elsewhere (cosine ~0)."""

    VECTORS = {
        "What details must be included in the record?": [1.0, 0.01, 0.0],
        "What details must be recorded?": [0.999, 0.045, 0.0],
        "Which standard governs the welding procedure?": [0.0, 0.0, 1.0],
    }

    def __call__(self, texts):
        return [self.VECTORS[t] for t in texts]


def test_synthetic_trio_two_paraphrases_and_control_drops_exactly_one():
    """Two paraphrased near-dupe questions (same doc, different gold facts so
    stage (a) does not touch them) + one unrelated control -> exactly one
    question-dedup drop, control survives untouched."""
    pairs = [
        _qa("Q1", "docA", "What details must be included in the record?",
            ["docA_F1", "docA_F2"], [0.95, 0.90]),
        _qa("Q2", "docA", "What details must be recorded?",
            ["docA_F3"], [0.80]),
        _qa("Q3", "docA", "Which standard governs the welding procedure?",
            ["docA_F4"], [0.85]),
    ]
    kept, stats = dedup_pairs(pairs, embed_fn=_FakeEmbedder(), cosine_threshold=0.92)

    assert stats["n_dupes_dropped_question"] == 1
    assert stats["n_dupes_dropped_evidence"] == 0
    assert len(kept) == 2
    kept_ids = {qa["qa_id"] for qa in kept}
    assert "Q3" in kept_ids
    # the higher-evidence-unit paraphrase (Q1, 2 clauses) survives over Q2 (1 clause)
    assert "Q1" in kept_ids
    assert "Q2" not in kept_ids


def test_identical_gold_fact_ids_collapse_keeping_more_evidence_units():
    pairs = [
        _qa("Q1", "docA", "question one", ["docA_F1", "docA_F2"], [0.9, 0.9]),
        _qa("Q2", "docA", "question two", ["docA_F2", "docA_F1"], [0.6]),
    ]
    kept, stats = dedup_pairs(pairs, embed_fn=None)

    assert stats["n_dupes_dropped_evidence"] == 1
    assert len(kept) == 1
    assert kept[0]["qa_id"] == "Q1"


def test_identical_evidence_tie_breaks_on_higher_min_clause_nli():
    pairs = [
        _qa("Q1", "docA", "q1", ["docA_F1", "docA_F2"], [0.9, 0.55]),
        _qa("Q2", "docA", "q2", ["docA_F1", "docA_F2"], [0.9, 0.80]),
    ]
    kept, stats = dedup_pairs(pairs, embed_fn=None)

    assert stats["n_dupes_dropped_evidence"] == 1
    assert len(kept) == 1
    assert kept[0]["qa_id"] == "Q2"


def test_shared_evidence_scoped_per_doc_not_collapsed_across_docs():
    pairs = [
        _qa("Q1", "docA", "q1", ["F1", "F2"], [0.9, 0.9]),
        _qa("Q2", "docB", "q2", ["F1", "F2"], [0.9, 0.9]),
    ]
    kept, stats = dedup_pairs(pairs, embed_fn=None)

    assert stats["n_dupes_dropped_evidence"] == 0
    assert len(kept) == 2


def test_no_embed_fn_skips_question_dedup_only():
    pairs = [
        _qa("Q1", "docA", "What details must be included in the record?",
            ["docA_F1"], [0.9]),
        _qa("Q2", "docA", "What details must be recorded?",
            ["docA_F2"], [0.9]),
    ]
    kept, stats = dedup_pairs(pairs, embed_fn=None)

    assert stats["n_dupes_dropped_question"] == 0
    assert len(kept) == 2
