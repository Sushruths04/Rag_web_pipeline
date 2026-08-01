"""Render the multi-model bake-off results as one self-contained HTML page.

    python scripts/build_bakeoff_html.py

Reads data/model_bakeoff/results.json, writes data/model_bakeoff/comparison.html.
Every question is one row block: the shared evidence on top, then every model's
question and answer side by side, all visible at once.
"""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, List

OUT_DIR = Path("data/model_bakeoff")
RESULTS_PATH = OUT_DIR / "results.json"
HTML_PATH = OUT_DIR / "comparison.html"

# Categorical slots 1-6 from the validated reference palette. Adjacent-pair
# CVD and normal-vision gates pass in both modes (validate_palette.js).
# Light mode flags three slots under 3:1 contrast, so every segment carries a
# direct label and a legend — identity is never colour-alone.
STEM_ORDER = ["What", "How", "Which", "Under", "Why", "Other"]
STEM_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
STEM_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def bucket_stem(stem: str) -> str:
    s = (stem or "").strip().capitalize()
    return s if s in STEM_ORDER[:-1] else "Other"


def _verdict(cell: dict) -> str:
    if cell.get("error"):
        return "error"
    if cell.get("reject_reason"):
        return "reject"
    return "pass"


def build_summary(payload: dict) -> str:
    models = payload["models"]
    results = payload["results"]
    usage = payload.get("usage", {})
    wall = payload.get("wall_sec", {})
    n = len(payload["chains"])

    rows = []
    stem_rows = []
    for m in models:
        cells = results[m["id"]]
        verdicts = Counter(_verdict(c) for c in cells)
        qs = [c["question"] for c in cells if c.get("question")]
        ans = [c["answer"] for c in cells if c.get("answer")]
        lat = [
            (c.get("qgen_sec") or 0) + (c.get("agen_sec") or 0)
            for c in cells if c.get("qgen_sec")
        ]
        u = usage.get(m["id"], {}) or {}
        rows.append(f"""
        <tr>
          <td class="mname">{esc(m['label'])}<span class="tier">{esc(m['tier'])}</span></td>
          <td class="num"><strong>{verdicts['pass']}</strong> / {n}</td>
          <td class="num">{verdicts['reject']}</td>
          <td class="num">{verdicts['error']}</td>
          <td class="num">{round(mean(len(q.split()) for q in qs), 1) if qs else '—'}</td>
          <td class="num">{round(mean(len(a.split()) for a in ans), 1) if ans else '—'}</td>
          <td class="num">{round(mean(lat), 1) if lat else '—'}s</td>
          <td class="num">{u.get('total_tokens', '—'):,}</td>
          <td class="num">{wall.get(m['id'], '—')}s</td>
        </tr>""")

        counts = Counter(bucket_stem(c.get("stem")) for c in cells if c.get("question"))
        total = sum(counts.values()) or 1
        segs, legend_used = [], []
        for i, name in enumerate(STEM_ORDER):
            v = counts.get(name, 0)
            if not v:
                continue
            pct = v / total * 100
            legend_used.append(name)
            # Direct label inside the segment when it is wide enough to hold
            # one; the relief rule for the light-mode contrast WARN.
            label = f'<span class="seglab">{name} {v}</span>' if pct >= 11 else ""
            segs.append(
                f'<div class="seg s{i}" style="width:{pct:.2f}%" '
                f'title="{esc(name)}: {v} of {total}">{label}</div>'
            )
        top = counts.most_common(1)[0] if counts else ("—", 0)
        stem_rows.append(f"""
        <div class="stemrow">
          <div class="stemname">{esc(m['label'])}</div>
          <div class="stembar">{''.join(segs)}</div>
          <div class="stemtop">{esc(top[0])} {round(top[1] / total * 100)}%</div>
        </div>""")

    legend = "".join(
        f'<span class="lg"><i class="s{i}"></i>{esc(nm)}</span>'
        for i, nm in enumerate(STEM_ORDER)
    )

    return f"""
    <section class="panel">
      <h2>Summary</h2>
      <p class="note">Gate pass = the pair would have survived
      <code>pipeline._reject_pair_reason</code>, the grounding check and the
      abstention check. Rejected pairs are still shown below so every model has a
      value in every row.</p>
      <div class="tablewrap">
        <table class="sum">
          <thead><tr>
            <th>Model</th><th>Gate pass</th><th>Rejected</th><th>Errors</th>
            <th>Q words</th><th>A words</th><th>Mean latency</th>
            <th>Tokens</th><th>Wall</th>
          </tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>

      <h3>Question-stem distribution</h3>
      <p class="note">Which interrogative each model reaches for. A column
      dominated by one stem is generating a narrow band of question types.</p>
      <div class="legend">{legend}</div>
      <div class="stems">{''.join(stem_rows)}</div>
    </section>"""


def build_rows(payload: dict) -> str:
    models = payload["models"]
    results = payload["results"]
    chains = payload["chains"]
    by_chain: Dict[str, Dict[str, dict]] = {}
    for m in models:
        for cell in results[m["id"]]:
            by_chain.setdefault(cell["chain_id"], {})[m["id"]] = cell

    blocks = []
    for idx, chain in enumerate(chains, 1):
        facts_html = "".join(
            f"""<div class="fact">
                  <span class="role">{esc(f.get('role', '—'))}</span>
                  <span class="pg">p.{esc(f.get('page_start', '?'))}</span>
                  <span class="ftext">{esc(f.get('text', ''))}</span>
                </div>"""
            for f in chain["facts"]
        )

        cols = []
        for m in models:
            cell = by_chain.get(chain["chain_id"], {}).get(m["id"], {})
            v = _verdict(cell)
            if v == "error":
                body = f'<div class="err">{esc(cell.get("error"))}</div>'
            else:
                flag = ""
                if cell.get("reject_reason"):
                    flag = (
                        f'<div class="flag">rejected: '
                        f'{esc(cell["reject_reason"])}</div>'
                    )
                if cell.get("reframed"):
                    flag += '<div class="flag reframe">reframed</div>'
                body = f"""
                  <div class="q">{esc(cell.get('question') or '—')}</div>
                  <div class="a">{esc(cell.get('answer') or '—')}</div>
                  {flag}
                  <div class="meta">{cell.get('qgen_sec') or '—'}s + {cell.get('agen_sec') or '—'}s</div>"""
            cols.append(f"""
              <div class="col v-{v}">
                <div class="colhead">{esc(m['label'])}<span class="dot"></span></div>
                {body}
              </div>""")

        kind = "2-fact chain" if chain["depth"] > 1 else "single fact"
        edge = ""
        if chain.get("chain_edges"):
            edge = f' · edge: {esc(chain["chain_edges"][0].get("type", "?"))}'
        blocks.append(f"""
        <article class="block">
          <header class="bhead">
            <span class="bnum">{idx:02d}</span>
            <span class="bmeta">{esc(chain['doc_id'])} · {kind}{edge}</span>
          </header>
          <div class="evidence">{facts_html}</div>
          <div class="grid">{''.join(cols)}</div>
        </article>""")
    return "".join(blocks)


CSS = """
:root{color-scheme:light dark;
 --surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
 --grid:#e1e0d9;--line:#c3c2b7;--ring:rgba(11,11,11,.10);
 --good:#0ca30c;--warn:#fab219;--crit:#d03b3b;
 --s0:#2a78d6;--s1:#eb6834;--s2:#1baf7a;--s3:#eda100;--s4:#e87ba4;--s5:#008300;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])){
 --surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--line:#383835;--ring:rgba(255,255,255,.10);
 --s0:#3987e5;--s1:#d95926;--s2:#199e70;--s3:#c98500;--s4:#d55181;--s5:#008300;}}
:root[data-theme=dark]{
 --surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--line:#383835;--ring:rgba(255,255,255,.10);
 --s0:#3987e5;--s1:#d95926;--s2:#199e70;--s3:#c98500;--s4:#d55181;--s5:#008300;}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
 font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;}
.wrap{max-width:1500px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:19px;margin:0 0 6px}
h3{font-size:15px;margin:26px 0 4px}
.sub{color:var(--ink2);margin:0 0 26px}
.note{color:var(--ink2);font-size:13px;margin:0 0 14px;max-width:78ch}
code{background:var(--grid);padding:1px 5px;border-radius:4px;font-size:12px}
.panel{background:var(--surface);border:1px solid var(--ring);border-radius:12px;
 padding:22px;margin-bottom:30px}
.tablewrap{overflow-x:auto}
table.sum{border-collapse:collapse;width:100%;font-size:13.5px;min-width:720px}
table.sum th{text-align:right;color:var(--muted);font-weight:600;font-size:12px;
 text-transform:uppercase;letter-spacing:.03em;padding:0 10px 8px;
 border-bottom:1px solid var(--line)}
table.sum th:first-child{text-align:left}
table.sum td{padding:10px;border-bottom:1px solid var(--grid)}
td.num{text-align:right;font-variant-numeric:tabular-nums}
td.mname{font-weight:600}
.tier{display:block;font-weight:400;font-size:11.5px;color:var(--muted)}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:10px 0 12px}
.lg{display:flex;align-items:center;gap:6px;font-size:12.5px;color:var(--ink2)}
.lg i{width:11px;height:11px;border-radius:3px;display:inline-block}
.stems{display:flex;flex-direction:column;gap:8px}
.stemrow{display:grid;grid-template-columns:150px 1fr 70px;gap:12px;align-items:center}
.stemname{font-size:13px;font-weight:600;text-align:right}
.stembar{display:flex;height:26px;border-radius:5px;overflow:hidden;gap:2px}
.stemtop{font-size:12px;color:var(--ink2);font-variant-numeric:tabular-nums}
.seg{display:flex;align-items:center;justify-content:center;min-width:3px}
.seglab{font-size:11px;font-weight:600;color:#fff;white-space:nowrap;
 padding:0 4px;text-shadow:0 1px 2px rgba(0,0,0,.4)}
.s0,i.s0{background:var(--s0)}.s1,i.s1{background:var(--s1)}
.s2,i.s2{background:var(--s2)}.s3,i.s3{background:var(--s3)}
.s4,i.s4{background:var(--s4)}.s5,i.s5{background:var(--s5)}
.block{background:var(--surface);border:1px solid var(--ring);border-radius:12px;
 padding:18px;margin-bottom:18px}
.bhead{display:flex;align-items:baseline;gap:12px;margin-bottom:12px}
.bnum{font-size:20px;font-weight:700;color:var(--muted);font-variant-numeric:tabular-nums}
.bmeta{font-size:12.5px;color:var(--ink2)}
.evidence{background:var(--plane);border-left:3px solid var(--line);
 border-radius:0 8px 8px 0;padding:12px 14px;margin-bottom:14px}
.fact{margin-bottom:8px;font-size:13.5px}
.fact:last-child{margin-bottom:0}
.role{display:inline-block;background:var(--grid);color:var(--ink2);font-size:10.5px;
 text-transform:uppercase;letter-spacing:.04em;padding:2px 7px;border-radius:4px;
 margin-right:7px;font-weight:600}
.pg{color:var(--muted);font-size:11.5px;margin-right:7px;font-variant-numeric:tabular-nums}
.ftext{color:var(--ink2)}
.grid{display:grid;grid-auto-flow:column;grid-auto-columns:minmax(270px,1fr);
 gap:12px;overflow-x:auto;padding-bottom:6px}
.col{border:1px solid var(--ring);border-radius:9px;padding:12px;background:var(--plane)}
.col.v-pass{border-left:3px solid var(--good)}
.col.v-reject{border-left:3px solid var(--warn)}
.col.v-error{border-left:3px solid var(--crit)}
.colhead{display:flex;align-items:center;gap:6px;font-size:12px;font-weight:700;
 color:var(--ink2);text-transform:uppercase;letter-spacing:.03em;margin-bottom:9px;
 padding-bottom:7px;border-bottom:1px solid var(--grid)}
.dot{width:8px;height:8px;border-radius:50%;margin-left:auto}
.v-pass .dot{background:var(--good)}.v-reject .dot{background:var(--warn)}
.v-error .dot{background:var(--crit)}
.q{font-size:13.5px;font-weight:600;margin-bottom:9px}
.a{font-size:13px;color:var(--ink2);padding-top:9px;border-top:1px dashed var(--grid)}
.flag{margin-top:9px;font-size:11px;font-weight:600;color:var(--crit);
 background:color-mix(in srgb,var(--warn) 16%,transparent);
 padding:3px 7px;border-radius:4px;display:inline-block}
.flag.reframe{color:var(--ink2);background:var(--grid)}
.err{font-size:12.5px;color:var(--crit);font-family:ui-monospace,monospace;
 word-break:break-word}
.meta{margin-top:9px;font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums}
"""


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS_PATH))
    ap.add_argument("--out", default=str(HTML_PATH))
    args = ap.parse_args()

    results_path, out_path = Path(args.results), Path(args.out)
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    n = len(payload["chains"])
    docs = sorted({c["doc_id"] for c in payload["chains"]})

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multi-model Q&amp;A bake-off — {n} questions × {len(payload['models'])} models</title>
<style>{CSS}</style></head>
<body><div class="wrap">
<h1>Multi-model Q&amp;A bake-off</h1>
<p class="sub">{n} fact chains from {esc(', '.join(docs))} · {len(payload['models'])} models ·
generated {esc(payload.get('generated_at', ''))}<br>
Every model saw the identical evidence. Only question generation and answer
generation vary.</p>
{build_summary(payload)}
<h2>All {n} questions</h2>
<p class="note">Each block shows the source facts, then every model's question and
answer for those same facts. Left border: green = passed the quality gate,
amber = generated but rejected, red = the call failed.</p>
{build_rows(payload)}
</div></body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    print(f"wrote {out_path}  ({len(page):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
