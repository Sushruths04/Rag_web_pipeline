from rag_gt.cli.adaptive_topk import adapt_retrieval
from rag_gt.core.types import Fact, MSFS, QuestionGT, Span


def _question(qid: str, chunk_ids: list[str]) -> QuestionGT:
    facts = [
        Fact(
            fact_id=f"{qid}_F{i}",
            text=f"Fact {i} about {qid}.",
            role="rule",
            supporting_spans=[
                Span(doc_id="doc", chunk_id=cid, start_token=0, end_token=4)
            ],
        )
        for i, cid in enumerate(chunk_ids)
    ]
    return QuestionGT(
        q_id=qid,
        question=f"{qid}?",
        gold_answer="A.",
        msfs_list=[MSFS(msfs_id=f"{qid}_m1", fact_ids=[f.fact_id for f in facts])],
        doc_ids=["doc"],
        required_fact_ids=[f.fact_id for f in facts],
        difficulty_reasoning_depth=2,
        difficulty_semantic_distance="intra_doc",
        required_facts=facts,
        required_fact_groups=[[f.fact_id] for f in facts],
    )


def test_adaptive_topk_keeps_top5_when_joint_coverage_already_met():
    q = _question("doc_q1", ["doc_c000001", "doc_c000003"])
    rows = [
        {
            "q_id": q.q_id,
            "retrieved_chunk_ids": [
                "doc_c000001",
                "doc_c000099",
                "doc_c000003",
                "doc_c000098",
                "doc_c000097",
                "doc_c000096",
            ],
        }
    ]

    out, decisions = adapt_retrieval([q], rows, ks=[5, 8, 10])

    assert out[0]["adaptive_topk"] == 5
    assert out[0]["adaptive_target_met"] is True
    assert out[0]["retrieved_chunk_ids"] == rows[0]["retrieved_chunk_ids"][:5]
    assert decisions[0]["adaptive_joint_fact_recall"] == 1.0


def test_adaptive_topk_expands_until_joint_coverage_is_met():
    q = _question("doc_q2", ["doc_c000001", "doc_c000006"])
    rows = [
        {
            "q_id": q.q_id,
            "retrieved_chunk_ids": [
                "doc_c000001",
                "doc_c000099",
                "doc_c000098",
                "doc_c000097",
                "doc_c000096",
                "doc_c000006",
                "doc_c000095",
                "doc_c000094",
            ],
        }
    ]

    out, decisions = adapt_retrieval([q], rows, ks=[5, 8, 10])

    assert out[0]["adaptive_topk"] == 8
    assert out[0]["adaptive_target_met"] is True
    assert out[0]["retrieved_chunk_ids"] == rows[0]["retrieved_chunk_ids"][:8]
    assert decisions[0]["adaptive_fact_recall"] == 1.0


def test_adaptive_topk_uses_max_k_when_target_is_not_met():
    q = _question("doc_q3", ["doc_c000001", "doc_c000010"])
    rows = [
        {
            "q_id": q.q_id,
            "retrieved_chunk_ids": [
                "doc_c000001",
                "doc_c000099",
                "doc_c000098",
                "doc_c000097",
                "doc_c000096",
                "doc_c000095",
                "doc_c000094",
                "doc_c000093",
                "doc_c000092",
                "doc_c000091",
            ],
        }
    ]

    out, decisions = adapt_retrieval([q], rows, ks=[5, 8, 10])

    assert out[0]["adaptive_topk"] == 10
    assert out[0]["adaptive_target_met"] is False
    assert out[0]["retrieved_chunk_ids"] == rows[0]["retrieved_chunk_ids"][:10]
    assert decisions[0]["adaptive_missed_fact_ids"] == ["doc_q3_F1"]
