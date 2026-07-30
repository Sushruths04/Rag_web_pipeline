"""Self-contained HTML evaluation report — no external assets, dark theme
matching the studio. Rendered by the Report tab iframe and downloadable.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone


def _esc(v) -> str:
    return html.escape(str(v))


_CSS = """
body { background:#0b0f0e; color:#e8f2ed; font-family:'Segoe UI',system-ui,sans-serif;
       font-size:14px; margin:0; padding:32px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:15px; margin:28px 0 10px; color:#8fa69d;
     text-transform:uppercase; letter-spacing:.08em; }
.sub { color:#8fa69d; margin-bottom:24px; }
.cards { display:flex; gap:12px; flex-wrap:wrap; }
.card { background:#121817; border:1px solid #223029; border-radius:10px; padding:14px 18px; min-width:130px; }
.card .v { font-size:22px; font-weight:700; color:#17e2b6; font-family:Consolas,monospace; }
.card .l { font-size:11px; color:#8fa69d; text-transform:uppercase; letter-spacing:.06em; margin-top:2px; }
table { border-collapse:collapse; width:100%; font-family:Consolas,monospace; font-size:12px; }
th { text-align:left; color:#5c6f68; font-size:10.5px; text-transform:uppercase; letter-spacing:.08em;
     padding:6px 10px; border-bottom:1px solid #223029; }
td { padding:6px 10px; border-bottom:1px solid #1a2522; }
tr.winner td { background:rgba(23,226,182,.08); }
.hit { color:#3ddc97; } .miss { color:#ff6b6b; }
details { margin:4px 0; } summary { cursor:pointer; }
.gold { color:#f5b84c; } .snippet { color:#8fa69d; font-size:11px; margin:2px 0 2px 18px; }
.note { background:#121817; border-left:3px solid #17e2b6; padding:12px 16px; border-radius:0 8px 8px 0;
        color:#8fa69d; line-height:1.6; margin-top:8px; }
"""


def build_report(
    *,
    run_id: str,
    docs: list[str],
    n_pairs: int,
    leaderboard: list[dict],
    eval_results: dict,
    gt_pairs: list[dict],
    chunk_texts: dict[str, str],
) -> str:
    agg = eval_results["aggregate"]
    winner = eval_results["winner"]
    by_id = {p.get("qa_id"): p for p in gt_pairs}

    cards = "".join(
        f'<div class="card"><div class="v">{_esc(v)}</div><div class="l">{_esc(l)}</div></div>'
        for l, v in [
            ("pairs evaluated", agg["n"]),
            ("recall", f"{agg['recall']:.3f}"),
            ("precision (rank-weighted)", f"{agg.get('precision_rw', agg['precision']):.3f}"),
            ("F1 (rank-weighted)", f"{agg.get('f1_rw', agg['f1']):.3f}"),
            ("raw precision@k", f"{agg['precision']:.3f}"),
            ("full-hit rate", f"{agg['hit_rate']:.3f}"),
            ("winning config", winner["config"]),
        ]
    )

    hop_rows = "".join(
        f"<tr><td>{_esc(hop)}</td><td>{d['n']}</td><td>{d['recall']:.3f}</td>"
        f"<td>{d.get('precision_rw', d['precision']):.3f}</td>"
        f"<td>{d['precision']:.3f}</td><td>{d['hit_rate']:.3f}</td></tr>"
        for hop, d in agg.get("by_hop", {}).items()
    )

    lb_rows = "".join(
        f'<tr class="{"winner" if r.get("winner") else ""}">'
        f"<td>{'★ ' if r.get('winner') else ''}{_esc(r['config'])}</td>"
        f"<td>{_esc(r['retrieval'])}</td><td>{r['top_k']}</td>"
        f"<td>{r['recall']:.3f}</td><td>{r.get('precision_rw', r['precision']):.3f}</td>"
        f"<td><b>{r.get('f1_rw', r['f1']):.3f}</b></td><td>{r['f1']:.3f}</td><td>{r['hit_rate']:.3f}</td></tr>"
        for r in sorted(leaderboard, key=lambda r: -r.get("f1_rw", r["f1"]))
    )

    q_rows = []
    for row in eval_results["rows"]:
        pair = by_id.get(row["qa_id"], {})
        units = [c.get("text", "") for c in (pair.get("answer_clauses") or [])] or [pair.get("answer", "")]
        gold_html = "".join(f'<div class="snippet gold">◆ {_esc(u)}</div>' for u in units)
        retrieved_html = "".join(
            f'<div class="snippet">{_esc(cid)} — {_esc(chunk_texts.get(cid, "")[:220])}…</div>'
            for cid in row["retrieved"]
        )
        cls = "hit" if row["hit"] else ("miss" if row["recall"] == 0 else "")
        q_rows.append(
            f"<details><summary><span class=\"{cls}\">"
            f"{'✓' if row['hit'] else '•' if row['recall'] > 0 else '✕'}</span> "
            f"[{_esc(row['hop_type'])}] recall {row['recall']:.2f} — "
            f"{_esc(pair.get('question', row['qa_id']))}</summary>"
            f"<div style='margin:6px 0 10px 18px'>"
            f"<div class='snippet'><b>Answer:</b> {_esc(pair.get('answer', ''))}</div>"
            f"<div style='margin-top:6px'><b style='font-size:11px'>Gold evidence</b>{gold_html}</div>"
            f"<div style='margin-top:6px'><b style='font-size:11px'>Retrieved ({winner['config']})</b>{retrieved_html}</div>"
            f"</div></details>"
        )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>GRAFT evaluation report — {_esc(run_id)}</title><style>{_CSS}</style></head><body>
<h1>Retrieval evaluation report</h1>
<div class="sub">run <b>{_esc(run_id)}</b> · corpus: {_esc(', '.join(docs))} · {ts}</div>
<div class="cards">{cards}</div>

<h2>Method</h2>
<div class="note">Every number on this page is computed <b>without an LLM judge</b>:
a ground-truth evidence unit counts as retrieved when at least 60% of its tokens
appear in a single retrieved chunk — the evidence text, pages, and bounding boxes
were captured when the ground truth was built, so verification is a deterministic
text-containment check. The same rule scores every retrieval configuration,
which is what makes the auto-tune leaderboard comparable, reproducible, and free.
Precision is reported <b>rank-weighted</b>: it credits ranking the gold evidence at
the top instead of demanding every retrieved chunk be gold — a 1-evidence question
retrieved at rank 1 scores 1.0, not 1/k. The raw precision@k column is kept for
transparency.</div>

<h2>Per-hop breakdown</h2>
<table><tr><th>hop type</th><th>n</th><th>recall</th><th>precision_rw</th><th>raw precision</th><th>full-hit</th></tr>{hop_rows}</table>

<h2>Auto-tune leaderboard ({len(leaderboard)} configurations, winner selected by rank-weighted F1)</h2>
<table><tr><th>config</th><th>retrieval</th><th>top-k</th><th>recall</th><th>precision_rw</th><th>F1_rw</th><th>raw F1</th><th>full-hit</th></tr>{lb_rows}</table>

<h2>Per-question drill-down ({n_pairs} pairs)</h2>
{''.join(q_rows)}
</body></html>"""
