"""Local HTML evidence pages with PDF page images and bbox highlights."""

from __future__ import annotations

import base64
import json
from html import escape
from pathlib import Path
from typing import Dict, Iterable


def write_fact_evidence_pages(
    gt_entries: Iterable[dict],
    output_dir: Path | str,
    *,
    max_pages: int = 1000,
) -> Dict[str, str]:
    """Write one HTML evidence page per fact with source bbox metadata.

    Returns ``fact_id -> relative html path`` from ``output_dir``.
    """
    out_dir = Path(output_dir)
    evidence_dir = out_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    links: Dict[str, str] = {}
    rendered = 0
    for q in gt_entries:
        for fact in q.get("required_facts") or []:
            fact_id = str(fact.get("fact_id", "") or "")
            spans = fact.get("supporting_spans") or []
            if not fact_id or not spans:
                continue
            span = spans[0]
            rel = f"evidence/{_safe_name(fact_id)}.html"
            target = out_dir / rel
            try:
                target.write_text(
                    _render_fact_page(q, fact, span),
                    encoding="utf-8",
                )
                rendered += 1
                links[fact_id] = rel
            except Exception as e:
                target.write_text(
                    _fallback_page(q, fact, span, f"{type(e).__name__}: {e}"),
                    encoding="utf-8",
                )
                links[fact_id] = rel
            if rendered >= max_pages:
                return links
    return links


def _render_fact_page(q: dict, fact: dict, span: dict) -> str:
    source_path = str(span.get("source_path", "") or "")
    bboxes = span.get("bboxes") or []
    page_no = int(span.get("page_start") or (bboxes[0].get("page_no") if bboxes else 0) or 0)
    if not source_path or not bboxes or page_no <= 0:
        return _fallback_page(q, fact, span, "No source_path/page/bbox metadata found.")

    import fitz

    with fitz.open(source_path) as pdf:
        page = pdf[page_no - 1]
        zoom = 1.5
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        image_b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
        width = pix.width
        height = pix.height
        page_height = float(page.rect.height)

    overlays = []
    for bbox in bboxes:
        if int(bbox.get("page_no", page_no) or page_no) != page_no:
            continue
        x0 = float(bbox.get("l", 0.0) or 0.0)
        x1 = float(bbox.get("r", 0.0) or 0.0)
        origin = str(bbox.get("coord_origin", "BOTTOMLEFT") or "BOTTOMLEFT").upper()
        if origin == "BOTTOMLEFT":
            y0 = page_height - float(bbox.get("t", 0.0) or 0.0)
            y1 = page_height - float(bbox.get("b", 0.0) or 0.0)
        else:
            y0 = float(bbox.get("t", 0.0) or 0.0)
            y1 = float(bbox.get("b", 0.0) or 0.0)
        left = min(x0, x1) * zoom
        top = min(y0, y1) * zoom
        w = abs(x1 - x0) * zoom
        h = abs(y1 - y0) * zoom
        overlays.append(
            f"<div class='box' style='left:{left:.2f}px;top:{top:.2f}px;"
            f"width:{w:.2f}px;height:{h:.2f}px'></div>"
        )

    return _page_shell(
        q=q,
        fact=fact,
        span=span,
        body=(
            f"<div class='pdf' style='width:{width}px;height:{height}px'>"
            f"<img src='data:image/png;base64,{image_b64}' width='{width}' height='{height}'>"
            f"{''.join(overlays)}</div>"
        ),
    )


def _fallback_page(q: dict, fact: dict, span: dict, reason: str) -> str:
    meta = json.dumps(span, indent=2, ensure_ascii=False)
    source_path = str(span.get("source_path", "") or "")
    page = span.get("page_start") or ""
    page_link = (
        f"<p><a href='file:///{escape(source_path)}#page={escape(str(page))}'>Open PDF page {escape(str(page))}</a></p>"
        if source_path and page else ""
    )
    return _page_shell(
        q=q,
        fact=fact,
        span=span,
        body=(
            f"<p class='warn'>{escape(reason)}</p>{page_link}"
            f"<pre>{escape(meta)}</pre>"
        ),
    )


def _page_shell(q: dict, fact: dict, span: dict, body: str) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{escape(str(fact.get('fact_id', 'evidence')))}</title>
<style>
body{{font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;background:#0d1117;color:#c9d1d9}}
.meta{{max-width:1100px;margin-bottom:16px}}
.muted{{color:#8b949e}} .warn{{color:#d29922}}
.pdf{{position:relative;background:#111;border:1px solid #30363d}}
.pdf img{{display:block}}
.box{{position:absolute;border:3px solid #f85149;background:rgba(248,81,73,.18);box-shadow:0 0 0 1px rgba(255,255,255,.4) inset}}
pre{{white-space:pre-wrap;background:#161b22;border:1px solid #30363d;padding:12px;border-radius:6px}}
</style></head><body>
<div class="meta">
<h1>{escape(str(fact.get('fact_id', '')))}</h1>
<p class="muted">q_id: {escape(str(q.get('q_id', '')))} · chunk: {escape(str(span.get('chunk_id', '')))} · page: {escape(str(span.get('page_start', '')))}</p>
<p><b>Question:</b> {escape(str(q.get('question', '')))}</p>
<p><b>Fact:</b> {escape(str(fact.get('text', '')))}</p>
</div>
{body}
</body></html>"""


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)
