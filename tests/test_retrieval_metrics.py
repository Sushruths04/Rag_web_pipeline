"""Tests for the facts-level retrieval evaluator."""

from __future__ import annotations

from rag_gt.comparison.retrieval_metrics import evaluate_corpus, evaluate_question
from rag_gt.comparison.chunk_resolver import ChunkRecord, ChunkResolver
from rag_gt.core.types import Fact, MSFS, QuestionGT, RetrievalLog, Span


def _make_q(qid: str, fact_chunks: list[list[str]], roles: list[str] | None = None) -> QuestionGT:
    facts = []
    for i, chunks in enumerate(fact_chunks):
        facts.append(
            Fact(
                fact_id=f"{qid}_F{i}",
                text=f"Fact {i} text long enough about {qid}.",
                role=(roles[i] if roles else "rule"),
                supporting_spans=[
                    Span(doc_id="doc", chunk_id=c, start_token=0, end_token=4)
                    for c in chunks
                ],
            )
        )
    return QuestionGT(
        q_id=qid,
        question=f"{qid}?",
        gold_answer="A.",
        msfs_list=[MSFS(msfs_id=f"{qid}_m1", fact_ids=[f.fact_id for f in facts])],
        doc_ids=["doc"],
        required_fact_ids=[f.fact_id for f in facts],
        difficulty_reasoning_depth=1,
        difficulty_semantic_distance="local",
        required_facts=facts,
    )


def test_per_question_perfect_recall():
    q = _make_q("doc_q1", [["doc_c000000"], ["doc_c000001"]])
    ret = RetrievalLog(q_id="doc_q1", retrieved_chunk_ids=["doc_c000000", "doc_c000001", "doc_c000099"])
    r = evaluate_question(q, ret)
    assert r.fact_recall == 1.0
    assert r.span_recall == 1.0
    # 2 of 3 retrieved chunks ground a required fact.
    assert r.fact_precision == 2 / 3
    assert r.joint_fact_recall == 1.0
    assert r.required_group_recall == 1.0
    assert round(r.fact_f1, 6) == round(2 * 1.0 * (2 / 3) / (1.0 + (2 / 3)), 6)
    assert r.overretrieval_penalty == 1 / 3
    assert r.hit_at_1 == 1
    assert r.hit_at_3 == 1
    assert r.mrr == (1 / 1 + 1 / 2) / 2  # F0 at rank 1, F1 at rank 2
    assert r.missed_fact_ids == []


def test_per_question_zero_recall():
    q = _make_q("doc_q2", [["doc_c000010"], ["doc_c000011"]])
    ret = RetrievalLog(q_id="doc_q2", retrieved_chunk_ids=["doc_c000099", "doc_c000098"])
    r = evaluate_question(q, ret)
    assert r.fact_recall == 0.0
    assert r.span_recall == 0.0
    assert r.fact_precision == 0.0
    assert r.hit_at_1 == 0
    assert r.mrr == 0.0
    assert sorted(r.missed_fact_ids) == ["doc_q2_F0", "doc_q2_F1"]


def test_partial_recall_and_normalisation():
    # GT uses 4-digit format, retrieval uses 6-digit — should still match.
    q = _make_q("doc_q3", [["doc_c0005"], ["doc_c0006"]])
    ret = RetrievalLog(q_id="doc_q3", retrieved_chunk_ids=["doc_c000005"])
    r = evaluate_question(q, ret)
    assert r.fact_recall == 0.5
    assert r.joint_fact_recall == 0.0
    assert r.required_group_recall == 0.0
    assert r.missed_fact_ids == ["doc_q3_F1"]


def test_required_group_recall_handles_independent_groups():
    q = _make_q("doc_q4", [["doc_c000001"], ["doc_c000002"]])
    q.required_fact_groups = [[q.required_facts[0].fact_id], [q.required_facts[1].fact_id]]
    ret = RetrievalLog(q_id="doc_q4", retrieved_chunk_ids=["doc_c000001", "doc_c000099"])
    r = evaluate_question(q, ret)
    assert r.fact_recall == 0.5
    assert r.joint_fact_recall == 0.0
    assert r.required_group_recall == 0.5


def test_source_span_recall_remaps_to_new_chunk_ids():
    q = QuestionGT(
        q_id="doc_q1",
        question="q",
        gold_answer="a",
        msfs_list=[MSFS(msfs_id="m1", fact_ids=["f1"])],
        doc_ids=["doc"],
        required_fact_ids=["f1"],
        difficulty_reasoning_depth=1,
        difficulty_semantic_distance="local",
        required_facts=[
            Fact(
                fact_id="f1",
                text="alpha beta gamma",
                role="definition",
                supporting_spans=[
                    Span(
                        doc_id="doc",
                        chunk_id="old_c000000",
                        start_token=0,
                        end_token=3,
                        char_start=10,
                        char_end=26,
                    )
                ],
            )
        ],
    )
    resolver = ChunkResolver(
        {
            "new_c000000": ChunkRecord(
                chunk_id="new_c000000",
                doc_id="doc",
                text="unrelated",
                char_start=0,
                char_end=9,
                sha1="x",
            ),
            "new_c000001": ChunkRecord(
                chunk_id="new_c000001",
                doc_id="doc",
                text="alpha beta gamma",
                char_start=9,
                char_end=40,
                sha1="y",
            ),
        }
    )

    r = evaluate_question(
        q,
        RetrievalLog(q_id="doc_q1", retrieved_chunk_ids=["new_c000001"]),
        resolver=resolver,
    )

    assert r.fact_recall == 1.0
    assert r.facts[0].required_chunk_ids == ["new_c1"]
    assert r.missed_fact_ids == []


def test_corpus_aggregation():
    q1 = _make_q("doc_q1", [["doc_c000000"]], roles=["rule"])
    q2 = _make_q("doc_q2", [["doc_c000010"]], roles=["definition"])
    q_map = {q1.q_id: q1, q2.q_id: q2}
    ret = {
        "doc_q1": RetrievalLog(q_id="doc_q1", retrieved_chunk_ids=["doc_c000000"]),
        "doc_q2": RetrievalLog(q_id="doc_q2", retrieved_chunk_ids=["doc_c000099"]),
    }
    c = evaluate_corpus(q_map, ret)
    assert c.n_questions == 2
    assert c.questions_full_recall == 1
    assert c.questions_zero_recall == 1
    assert c.fact_miss_rate == 0.5
    assert c.per_role_recall == {"definition": 0.0, "rule": 1.0}
    assert c.means["fact_recall"] == 0.5
