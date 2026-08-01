"""Consolidate per-document Q&A checkpoints into one dataset + an HTML viewer.

The pipeline writes pairs per document into
``<out-dir>/<doc>/checkpoints/s7_pairs.json``, which is fine for resuming a run
but useless for reading. This produces:

    <dest>/gt_pairs.jsonl   one pair per line, all documents, with doc_id
    <dest>/gt_pairs.json    the same as a single JSON array
    <dest>/index.html       self-contained, searchable, filterable viewer

Usage:
    python scripts/consolidate_gt.py data/reruns_postfix data/gt_final
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path


def load_runs(root: Path) -> list[dict]:
    pairs: list[dict] = []
    for doc_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cp = doc_dir / "checkpoints" / "s7_pairs.json"
        if not cp.exists():
            continue
        rows = json.loads(cp.read_text(encoding="utf-8"))
        for i, r in enumerate(rows, 1):
            r["doc_id"] = doc_dir.name
            r["pair_id"] = f"{doc_dir.name}_p{i:04d}"
            pairs.append(r)
    return pairs


def render(pairs: list[dict]) -> str:
    docs = sorted({p["doc_id"] for p in pairs})
    n_multi = sum(1 for p in pairs if (p.get("depth") or 1) > 1)

    rows = []
    for p in pairs:
        facts = p.get("facts") or []
        fact_html = "".join(
            f'<div class="fact"><span class="fid">{html.escape(str(f.get("fact_id","")))}'
            f'</span> <span class="pg">p.{f.get("page_start","?")}</span>'
            f'<div>{html.escape(str(f.get("text","")))}</div></div>'
            for f in facts
        )
        depth = p.get("depth") or 1
        badge = f'<span class="badge multi">{depth}-hop</span>' if depth > 1 else \
                '<span class="badge single">single</span>'
        nec = p.get("necessity_score")
        nec_html = f'<span class="badge nec">necessity {nec:.2f}</span>' if isinstance(nec, (int, float)) else ""
        rows.append(f"""
<article class="pair" data-doc="{html.escape(p['doc_id'])}" data-depth="{depth}">
  <header>
    <span class="pid">{html.escape(p['pair_id'])}</span>
    {badge}{nec_html}
    <span class="badge doc">{html.escape(p['doc_id'])}</span>
  </header>
  <div class="q"><b>Q</b> {html.escape(str(p.get('question','')))}</div>
  <div class="a"><b>A</b> {html.escape(str(p.get('answer','')))}</div>
  <details><summary>supporting facts ({len(facts)})</summary>{fact_html}</details>
</article>""")

    opts = "".join(f'<option value="{html.escape(d)}">{html.escape(d)}</option>' for d in docs)
    return f"""<!doctype html>
<meta charset="utf-8">
<title>GRAFT ground truth — {len(pairs)} pairs</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.5 system-ui, sans-serif; margin: 0; padding: 24px;
          max-width: 1000px; margin-inline: auto; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .sub {{ opacity: .7; margin-bottom: 18px; }}
  .controls {{ position: sticky; top: 0; padding: 12px 0; display: flex; gap: 10px;
               flex-wrap: wrap; background: Canvas; border-bottom: 1px solid #8884;
               margin-bottom: 16px; z-index: 5; }}
  input, select {{ padding: 6px 8px; font: inherit; }}
  input[type=search] {{ flex: 1; min-width: 220px; }}
  .pair {{ border: 1px solid #8884; border-radius: 8px; padding: 12px 14px;
           margin-bottom: 10px; }}
  header {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
            margin-bottom: 8px; }}
  .pid {{ font-family: ui-monospace, monospace; font-size: 11px; opacity: .6; }}
  .badge {{ font-size: 11px; padding: 2px 7px; border-radius: 99px; border: 1px solid #8886; }}
  .multi {{ background: #7c3aed22; border-color: #7c3aed88; }}
  .single {{ opacity: .65; }}
  .nec {{ background: #05966922; border-color: #05966988; }}
  .doc {{ margin-left: auto; opacity: .7; font-family: ui-monospace, monospace; }}
  .q {{ margin-bottom: 6px; }}
  .a {{ opacity: .85; }}
  b {{ display: inline-block; width: 16px; opacity: .5; }}
  details {{ margin-top: 8px; }}
  summary {{ cursor: pointer; font-size: 12px; opacity: .65; }}
  .fact {{ border-left: 2px solid #8886; padding: 4px 0 4px 10px; margin: 8px 0;
           font-size: 13px; }}
  .fid {{ font-family: ui-monospace, monospace; font-size: 11px; opacity: .55; }}
  .pg {{ font-size: 11px; opacity: .55; }}
  #count {{ align-self: center; font-size: 12px; opacity: .7; }}
</style>
<h1>GRAFT ground truth</h1>
<div class="sub">{len(pairs)} pairs across {len(docs)} documents · {n_multi} multi-hop</div>
<div class="controls">
  <input type="search" id="q" placeholder="search question, answer or fact text…">
  <select id="doc"><option value="">all documents</option>{opts}</select>
  <select id="hop">
    <option value="">all types</option>
    <option value="single">single-fact</option>
    <option value="multi">multi-hop</option>
  </select>
  <span id="count"></span>
</div>
<main id="list">{''.join(rows)}</main>
<script>
  const q = document.getElementById('q'), doc = document.getElementById('doc'),
        hop = document.getElementById('hop'), count = document.getElementById('count'),
        items = [...document.querySelectorAll('.pair')];
  function apply() {{
    const t = q.value.toLowerCase(), d = doc.value, h = hop.value;
    let n = 0;
    for (const el of items) {{
      const depth = +el.dataset.depth;
      const ok = (!d || el.dataset.doc === d)
        && (!h || (h === 'multi' ? depth > 1 : depth === 1))
        && (!t || el.textContent.toLowerCase().includes(t));
      el.hidden = !ok;
      if (ok) n++;
    }}
    count.textContent = n + ' shown';
  }}
  [q, doc, hop].forEach(el => el.addEventListener('input', apply));
  apply();
</script>
"""


def main(src: str, dest: str) -> None:
    root, out = Path(src), Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    pairs = load_runs(root)
    if not pairs:
        raise SystemExit(f"no s7_pairs.json found under {root}")

    (out / "gt_pairs.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + "\n",
        encoding="utf-8",
    )
    (out / "gt_pairs.json").write_text(
        json.dumps(pairs, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (out / "index.html").write_text(render(pairs), encoding="utf-8")

    by_doc: dict[str, int] = {}
    for p in pairs:
        by_doc[p["doc_id"]] = by_doc.get(p["doc_id"], 0) + 1
    print(f"{len(pairs)} pairs -> {out}")
    for d, n in sorted(by_doc.items()):
        print(f"  {d:20s} {n:5d}")
    print(f"  multi-hop: {sum(1 for p in pairs if (p.get('depth') or 1) > 1)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "data/gt_final")
