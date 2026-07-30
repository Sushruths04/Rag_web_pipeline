"""Block: report [FREE] -- eval (multi-in) -> report.

Backing: rag_gt.rag.eval_v2._fmt_md's row-formatting primitives (_row,
_HEAD_KEYS), reused verbatim (05_BLOCK_CATALOG.md §3.30). _fmt_md itself
formats the nested {"strategies": {...}} shape evaluate_v2's multi-strategy
CLI sweep produces; this block's "eval" port is a flat multi-in list of
independently produced eval artifacts (one per rag_gt.blocks.evaluator run),
so it is the "equivalent thin formatter" the catalog explicitly allows in
place of _fmt_md, built from the same real row-rendering function.
"""
from __future__ import annotations

from pathlib import Path

from rag_gt.blocks._common import artifact, read_json_artifact, write_text_artifact
from rag_gt.rag.eval_v2 import _HEAD_KEYS, _row

_MD_HEADER = (
    "| set | n | recall@5 | precision@5 | precision_rw@5 | hit@5 | mrr | coverage |\n"
    "|---|---:|---:|---:|---:|---:|---:|---:|"
)


def _label_for(index: int, eval_artifact: dict) -> str:
    meta = eval_artifact.get("meta", {}) or {}
    return str(meta.get("label") or meta.get("strategy") or f"eval_{index + 1}")


def _rows_from_inputs(eval_artifacts: list) -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    for i, art in enumerate(eval_artifacts):
        payload = read_json_artifact(art["ref"])
        rows.append((_label_for(i, art), payload["summary"]))
    return rows


def _render_md(rows: list[tuple[str, dict]]) -> str:
    lines = ["# Retrieval evaluation report", "", _MD_HEADER]
    for label, summary in rows:
        lines.append(_row(label, summary))
    return "\n".join(lines) + "\n"


def _render_html(rows: list[tuple[str, dict]]) -> str:
    header_cols = ("set", "n", "recall@5", "precision@5", "precision_rw@5", "hit@5", "mrr", "coverage")
    header = "".join(f"<th>{c}</th>" for c in header_cols)
    body = []
    for label, summary in rows:
        cells = [label, str(summary.get("n_pairs", 0))]
        cells += [f"{summary.get(key, 0.0):.3f}" for key in _HEAD_KEYS[1:]]
        body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return (
        "<table><thead><tr>" + header + "</tr></thead><tbody>"
        + "".join(body) + "</tbody></table>"
    )


def run(inputs: dict, params: dict, artifacts_dir: Path | str | None = None) -> dict:
    eval_artifacts = inputs.get("eval") or []
    fmt = str(params.get("format") or "html")
    rows = _rows_from_inputs(eval_artifacts)

    if fmt == "md":
        text = _render_md(rows)
    elif fmt == "html":
        text = _render_html(rows)
    else:
        raise ValueError(f"Unknown report format: {fmt!r}. Choose from: md, html")

    ref = write_text_artifact(artifacts_dir, "report", text, fmt)
    return {
        "report": artifact(
            "report", str(ref), {"path": str(ref), "n_inputs": len(eval_artifacts), "format": fmt}
        )
    }
