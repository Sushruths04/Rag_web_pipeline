"""TDD: rag_gt.blocks.report -- eval (multi-in) -> report
(05_BLOCK_CATALOG.md §3.30). Backing: rag_gt.rag.eval_v2._fmt_md's row
formatting primitives (_row/_HEAD_KEYS), reused verbatim -- report itself is
the "equivalent thin formatter" the catalog explicitly allows in place of
_fmt_md's single-strategy nested-dict shape, since this block's eval input is
a flat multi-in list of independently produced eval artifacts rather than
evaluate_v2's per-strategy sweep result.
"""
from __future__ import annotations

from rag_gt.rag.eval_v2 import _row


def _eval_artifact(tmp_path, name, summary):
    from rag_gt.blocks._common import artifact, write_json_artifact

    ref = write_json_artifact(tmp_path, f"evaluator_{name}", {"top_k": 5, "summary": summary, "by_pair_type": {}, "rows": []})
    return artifact("eval", str(ref), {**summary, "top_k": 5, "label": name})


SUMMARY_A = {"n_pairs": 2, "mrr": 1.0, "coverage": 1.0, "fact_recall@5": 1.0,
             "precision@5": 0.2, "precision_rw@5": 1.0, "hit@5": 1.0, "ndcg@5": 1.0}
SUMMARY_B = {"n_pairs": 3, "mrr": 0.5, "coverage": 0.5, "fact_recall@5": 0.5,
             "precision@5": 0.1, "precision_rw@5": 0.5, "hit@5": 0.0, "ndcg@5": 0.5}


def test_report_md_contains_row_per_eval_input(tmp_path):
    from rag_gt.blocks.report import run

    inputs = {"eval": [_eval_artifact(tmp_path, "bm25", SUMMARY_A), _eval_artifact(tmp_path, "hybrid", SUMMARY_B)]}
    out = run(inputs, {"format": "md"}, artifacts_dir=tmp_path)

    assert set(out.keys()) == {"report"}
    artifact = out["report"]
    assert artifact["type"] == "report"
    assert artifact["meta"]["n_inputs"] == 2
    assert artifact["ref"].endswith(".md")

    text = open(artifact["ref"], encoding="utf-8").read()
    assert _row("bm25", SUMMARY_A) in text
    assert _row("hybrid", SUMMARY_B) in text


def test_report_html_format(tmp_path):
    from rag_gt.blocks.report import run

    inputs = {"eval": [_eval_artifact(tmp_path, "bm25", SUMMARY_A)]}
    out = run(inputs, {"format": "html"}, artifacts_dir=tmp_path)

    assert out["report"]["ref"].endswith(".html")
    text = open(out["report"]["ref"], encoding="utf-8").read()
    assert "<table" in text
    assert "1.000" in text  # mrr formatted like _row does


def test_report_empty_eval_input_produces_header_only(tmp_path):
    from rag_gt.blocks.report import run

    out = run({"eval": []}, {"format": "md"}, artifacts_dir=tmp_path)
    assert out["report"]["meta"]["n_inputs"] == 0
