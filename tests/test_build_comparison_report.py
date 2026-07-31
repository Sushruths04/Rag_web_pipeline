from rag_gt.cli.build_comparison_report import (
    GoldFact,
    QView,
    _render_html,
    _render_markdown,
    _summarize_answers,
)


def _sample_view() -> QView:
    return QView(
        q_id="doc_q001",
        question="What is the key result?",
        gold_answer="The key result is alpha.",
        predicted_answer="[not generated in this retrieval-only V11 comparison]",
        gold_facts=[GoldFact(fact_id="F1", text="alpha", chunk_id="doc_c000001")],
        gold_chunk_ids=["doc_c000001"],
        retrieved_chunk_ids=["doc_c000001"],
        retrieved_chunks=[{"chunk_id": "doc_c000001", "text": "alpha"}],
        rag_gt={
            "strict_recall_l13": 1.0,
            "text_recall_l3": 1.0,
            "text_recall_l2": 1.0,
            "text_recall_any": 1.0,
            "fact_recall_l1": 1.0,
            "fact_precision_rw": 1.0,
        },
        ragas={"context_recall": 0.0, "context_precision": 0.0},
        diff_recall=1.0,
        diff_recall_strict=1.0,
        diff_precision=1.0,
        difficulty_depth=1,
        difficulty_distance="local",
    )


def test_answer_audit_detects_retrieval_only_mode():
    audit = _summarize_answers(
        [
            {
                "q_id": "doc_q001",
                "predicted_answer": "[not generated in this retrieval-only V11 comparison]",
                "abstained": True,
            }
        ]
    )
    assert audit.mode == "retrieval_only"
    assert audit.real_answers == 0
    assert audit.placeholders == 1
    assert audit.abstained == 1


def test_report_render_marks_retrieval_only_runs():
    audit = _summarize_answers(
        [
            {
                "q_id": "doc_q001",
                "predicted_answer": "[not generated in this retrieval-only V11 comparison]",
                "abstained": True,
            }
        ]
    )
    view = _sample_view()
    corpus = {
        "n_questions": 1,
        "speedup_rag_gt_over_ragas": 2.0,
        "rag_gt_seconds": 1.0,
        "ragas_seconds": 2.0,
        "ragas_usd": 0.1,
        "ragas_judge_calls": 3,
        "judge_model": "test-model",
        "ragas_tokens_in": 10,
        "ragas_tokens_out": 5,
    }

    html = _render_html(corpus, [view], [], audit)
    md = _render_markdown(corpus, [view], [], audit)

    assert "Mode: <b>retrieval-only</b>" in html
    assert "No SUT predicted answer was generated for this question." in html
    assert "**Report mode:** retrieval-only." in md
    assert "No SUT predicted answer was generated for this question." in md
