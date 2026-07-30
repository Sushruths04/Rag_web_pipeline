"""Build an interactive HTML dashboard from RAG-GT trace JSONL logs."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rag_gt.observability.tracing import PIPELINE_STAGE_MAP


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return events


def _summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    stage_counts = Counter(e.get("stage", "unknown") for e in events)
    status_counts = Counter(e.get("status", "unknown") for e in events)
    drop_reasons = Counter(
        e.get("reason") or "unspecified"
        for e in events
        if e.get("status") == "dropped" or e.get("event") == "drop"
    )
    drop_by_stage: dict[str, Counter] = defaultdict(Counter)
    durations: dict[str, float] = defaultdict(float)
    docs = sorted({e.get("doc_id") for e in events if e.get("doc_id")})
    item_events = [
        e for e in events
        if e.get("item_id") or e.get("event") in {"candidate_accepted", "drop"}
    ]
    for e in events:
        if e.get("status") == "dropped" or e.get("event") == "drop":
            drop_by_stage[e.get("stage", "unknown")][e.get("reason") or "unspecified"] += 1
        if e.get("event") == "stage_end":
            durations[e.get("stage", "unknown")] += float(
                (e.get("metrics") or {}).get("duration_ms", 0) or 0
            )
    accepted = [
        e for e in events
        if e.get("event") in {"candidate_accepted", "question_saved"}
        or e.get("status") == "accepted"
    ]
    return {
        "stage_counts": dict(stage_counts),
        "status_counts": dict(status_counts),
        "drop_reasons": dict(drop_reasons),
        "drop_by_stage": {k: dict(v) for k, v in drop_by_stage.items()},
        "stage_durations_ms": dict(durations),
        "docs": docs,
        "item_event_count": len(item_events),
        "accepted_count": len(accepted),
        "dropped_count": sum(drop_reasons.values()),
        "event_count": len(events),
    }


def _html_template(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    title = html.escape(data.get("title", "RAG-GT Pipeline Trace"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --bg: #f7f8fa;
  --panel: #ffffff;
  --ink: #1d2433;
  --muted: #667085;
  --line: #d6dbe3;
  --accent: #2662d9;
  --good: #1b8a5a;
  --bad: #c0392b;
  --warn: #b7791f;
  --shadow: 0 1px 2px rgba(16,24,40,.08);
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: Inter, Segoe UI, Arial, sans-serif; }}
header {{ position: sticky; top: 0; z-index: 5; background: rgba(255,255,255,.96); border-bottom: 1px solid var(--line); }}
.top {{ display: flex; gap: 16px; align-items: center; padding: 14px 20px; }}
h1 {{ font-size: 19px; margin: 0; font-weight: 650; }}
.sub {{ color: var(--muted); font-size: 12px; }}
.wrap {{ padding: 18px 20px 28px; max-width: 1500px; margin: 0 auto; }}
.controls {{ display: grid; grid-template-columns: 1.2fr 220px 220px 160px; gap: 10px; margin-bottom: 14px; }}
input, select, button {{ border: 1px solid var(--line); background: #fff; color: var(--ink); border-radius: 6px; padding: 9px 10px; font-size: 13px; }}
button {{ cursor: pointer; background: var(--accent); color: #fff; border-color: var(--accent); }}
.stats {{ display: grid; grid-template-columns: repeat(5, minmax(130px,1fr)); gap: 10px; margin-bottom: 16px; }}
.stat, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); }}
.stat {{ padding: 12px; }}
.stat b {{ display: block; font-size: 22px; margin-bottom: 3px; }}
.stat span {{ color: var(--muted); font-size: 12px; }}
.grid {{ display: grid; grid-template-columns: 1.2fr .8fr; gap: 14px; align-items: start; }}
.panel {{ padding: 13px; margin-bottom: 14px; }}
.panel h2 {{ margin: 0 0 10px; font-size: 15px; }}
.flow {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 8px; }}
.node {{ border: 1px solid var(--line); border-radius: 7px; background: #fbfcfe; padding: 9px; min-height: 82px; cursor: pointer; }}
.node.active {{ outline: 2px solid var(--accent); }}
.node .id {{ color: var(--accent); font-size: 11px; font-weight: 650; text-transform: uppercase; }}
.node .label {{ font-size: 13px; font-weight: 650; margin: 3px 0; }}
.node .meta {{ color: var(--muted); font-size: 11px; line-height: 1.3; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
th, td {{ padding: 7px 8px; border-bottom: 1px solid #eef1f5; text-align: left; vertical-align: top; }}
th {{ color: var(--muted); font-weight: 650; background: #fbfcfe; position: sticky; top: 55px; }}
tr:hover {{ background: #f6f8fb; }}
.badge {{ display: inline-block; padding: 2px 6px; border-radius: 999px; font-size: 11px; border: 1px solid var(--line); }}
.ok {{ color: var(--good); border-color: rgba(27,138,90,.35); background: rgba(27,138,90,.08); }}
.drop {{ color: var(--bad); border-color: rgba(192,57,43,.35); background: rgba(192,57,43,.08); }}
.error {{ color: var(--bad); }}
.bar {{ height: 8px; background: #e9edf4; border-radius: 999px; overflow: hidden; min-width: 90px; }}
.bar > i {{ display: block; height: 100%; background: var(--accent); }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #111827; color: #e5e7eb; padding: 11px; border-radius: 7px; max-height: 480px; overflow: auto; font-size: 12px; }}
.split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
.muted {{ color: var(--muted); }}
@media (max-width: 900px) {{
  .controls, .grid, .split {{ grid-template-columns: 1fr; }}
  .stats {{ grid-template-columns: repeat(2, minmax(120px,1fr)); }}
  th {{ position: static; }}
}}
</style>
</head>
<body>
<header>
  <div class="top">
    <div>
      <h1>{title}</h1>
      <div class="sub" id="subtitle"></div>
    </div>
  </div>
</header>
<main class="wrap">
  <div class="controls">
    <input id="search" placeholder="Search q_id, chain_id, fact_id, question, reason, stage">
    <select id="docFilter"><option value="">All documents</option></select>
    <select id="stageFilter"><option value="">All stages</option></select>
    <select id="statusFilter"><option value="">All statuses</option></select>
  </div>
  <section class="stats" id="stats"></section>
  <section class="grid">
    <div>
      <div class="panel">
        <h2>Pipeline Flow</h2>
        <div class="flow" id="flow"></div>
      </div>
      <div class="panel">
        <h2>Trace Events</h2>
        <table id="eventsTable">
          <thead><tr><th>Time</th><th>Stage</th><th>Status</th><th>Item</th><th>Reason / Detail</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
    <div>
      <div class="panel">
        <h2>Accepted vs Rejected</h2>
        <div class="split">
          <div><b>Accepted</b><table id="acceptedTable"><tbody></tbody></table></div>
          <div><b>Rejected</b><table id="rejectedTable"><tbody></tbody></table></div>
        </div>
      </div>
      <div class="panel">
        <h2>Drops by Reason</h2>
        <table id="dropsTable"><tbody></tbody></table>
      </div>
      <div class="panel">
        <h2>Bottlenecks</h2>
        <table id="durationTable"><tbody></tbody></table>
      </div>
      <div class="panel">
        <h2>Selected Event</h2>
        <pre id="detail">Select a row to inspect the full structured event.</pre>
      </div>
    </div>
  </section>
</main>
<script>
const DATA = {payload};
const events = DATA.events || [];
const summary = DATA.summary || {{}};
const stages = DATA.pipeline_stage_map || [];
let selectedStage = "";

function byId(id) {{ return document.getElementById(id); }}
function esc(v) {{ return String(v ?? "").replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
function eventText(e) {{ return JSON.stringify(e).toLowerCase(); }}
function statusBadge(s) {{
  const cls = s === "dropped" ? "drop" : (s === "error" ? "drop" : "ok");
  return `<span class="badge ${{cls}}">${{esc(s || "ok")}}</span>`;
}}
function fillFilters() {{
  const docs = [...new Set(events.map(e => e.doc_id).filter(Boolean))].sort();
  const stageIds = [...new Set(events.map(e => e.stage).filter(Boolean))].sort();
  const statuses = [...new Set(events.map(e => e.status).filter(Boolean))].sort();
  for (const d of docs) byId("docFilter").insertAdjacentHTML("beforeend", `<option>${{esc(d)}}</option>`);
  for (const s of stageIds) byId("stageFilter").insertAdjacentHTML("beforeend", `<option>${{esc(s)}}</option>`);
  for (const s of statuses) byId("statusFilter").insertAdjacentHTML("beforeend", `<option>${{esc(s)}}</option>`);
}}
function renderStats() {{
  byId("subtitle").textContent = `${{DATA.trace_path || ""}} · ${{events.length}} events`;
  const stats = [
    ["Events", summary.event_count ?? events.length],
    ["Documents", (summary.docs || []).length],
    ["Accepted", summary.accepted_count || 0],
    ["Dropped", summary.dropped_count || 0],
    ["Item events", summary.item_event_count || 0],
  ];
  byId("stats").innerHTML = stats.map(([k,v]) => `<div class="stat"><b>${{esc(v)}}</b><span>${{esc(k)}}</span></div>`).join("");
}}
function renderFlow() {{
  const counts = summary.stage_counts || {{}};
  byId("flow").innerHTML = stages.map(s => {{
    const n = counts[s.id] || 0;
    const active = selectedStage === s.id ? " active" : "";
    const outputs = (s.outputs || []).slice(0,3).join(", ");
    return `<div class="node${{active}}" data-stage="${{esc(s.id)}}">
      <div class="id">${{esc(s.id)}}</div>
      <div class="label">${{esc(s.label)}}</div>
      <div class="meta">${{n}} events<br>${{esc(outputs)}}</div>
    </div>`;
  }}).join("");
  document.querySelectorAll(".node").forEach(n => n.onclick = () => {{
    selectedStage = selectedStage === n.dataset.stage ? "" : n.dataset.stage;
    byId("stageFilter").value = selectedStage;
    renderAll();
  }});
}}
function filteredEvents() {{
  const q = byId("search").value.trim().toLowerCase();
  const doc = byId("docFilter").value;
  const stage = byId("stageFilter").value;
  const status = byId("statusFilter").value;
  return events.filter(e => (!q || eventText(e).includes(q)) && (!doc || e.doc_id === doc) && (!stage || e.stage === stage) && (!status || e.status === status));
}}
function reasonDetail(e) {{
  const d = e.data || {{}};
  const parts = [];
  if (e.reason) parts.push(e.reason);
  if (d.question) parts.push(d.question);
  if (d.q_id) parts.push(d.q_id);
  if (d.quality) parts.push(`quality=${{d.quality.quality ?? d.quality}}`);
  if (d.chain && d.chain.fact_ids) parts.push(d.chain.fact_ids.join(" → "));
  return parts.join(" · ") || e.event || "";
}}
function renderEvents() {{
  const rows = filteredEvents().slice(-700).reverse();
  byId("eventsTable").querySelector("tbody").innerHTML = rows.map((e, i) => `
    <tr data-idx="${{events.indexOf(e)}}">
      <td>${{esc(Math.round(e.elapsed_ms || 0))}}ms</td>
      <td>${{esc(e.stage)}}<div class="muted">${{esc(e.event)}}</div></td>
      <td>${{statusBadge(e.status)}}</td>
      <td>${{esc(e.item_id || "")}}<div class="muted">${{esc(e.doc_id || "")}}</div></td>
      <td>${{esc(reasonDetail(e))}}</td>
    </tr>`).join("");
  byId("eventsTable").querySelectorAll("tr[data-idx]").forEach(row => row.onclick = () => {{
    byId("detail").textContent = JSON.stringify(events[Number(row.dataset.idx)], null, 2);
  }});
}}
function renderAcceptedRejected() {{
  const evs = filteredEvents();
  const accepted = evs.filter(e => e.event === "candidate_accepted" || e.status === "accepted").slice(-25).reverse();
  const rejected = evs.filter(e => e.status === "dropped").slice(-25).reverse();
  byId("acceptedTable").querySelector("tbody").innerHTML = accepted.map(e => `<tr><td>${{esc(e.item_id || (e.data||{{}}).q_id || "")}}</td><td>${{esc((e.data||{{}}).question || e.stage)}}</td></tr>`).join("");
  byId("rejectedTable").querySelector("tbody").innerHTML = rejected.map(e => `<tr><td>${{esc(e.reason || "")}}</td><td>${{esc((e.data||{{}}).question || e.item_id || e.stage)}}</td></tr>`).join("");
}}
function renderDrops() {{
  const drops = Object.entries(summary.drop_reasons || {{}}).sort((a,b) => b[1]-a[1]);
  const max = Math.max(1, ...drops.map(x => x[1]));
  byId("dropsTable").querySelector("tbody").innerHTML = drops.map(([reason, n]) => `<tr><td>${{esc(reason)}}</td><td>${{n}}</td><td><div class="bar"><i style="width:${{100*n/max}}%"></i></div></td></tr>`).join("");
}}
function renderDurations() {{
  const durs = Object.entries(summary.stage_durations_ms || {{}}).sort((a,b) => b[1]-a[1]);
  const max = Math.max(1, ...durs.map(x => x[1]));
  byId("durationTable").querySelector("tbody").innerHTML = durs.map(([stage, ms]) => `<tr><td>${{esc(stage)}}</td><td>${{Math.round(ms)}}ms</td><td><div class="bar"><i style="width:${{100*ms/max}}%"></i></div></td></tr>`).join("");
}}
function renderAll() {{
  selectedStage = byId("stageFilter").value;
  renderStats(); renderFlow(); renderEvents(); renderAcceptedRejected(); renderDrops(); renderDurations();
}}
fillFilters();
["search","docFilter","stageFilter","statusFilter"].forEach(id => byId(id).addEventListener("input", renderAll));
renderAll();
</script>
</body>
</html>"""


def build_dashboard(trace_path: Path, output_path: Path) -> Path:
    events = _read_jsonl(trace_path)
    summary = _summarize(events)
    data = {
        "title": f"RAG-GT Pipeline Trace: {trace_path.name}",
        "trace_path": str(trace_path),
        "summary": summary,
        "pipeline_stage_map": PIPELINE_STAGE_MAP,
        "events": events,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_html_template(data), encoding="utf-8")
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag-gt-build-pipeline-dashboard",
        description="Build an interactive HTML dashboard from a RAG-GT trace JSONL file.",
    )
    p.add_argument("--trace", required=True, help="Path to *.trace.jsonl")
    p.add_argument("--output", default=None, help="Output HTML path")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    trace_path = Path(args.trace)
    output_path = Path(args.output) if args.output else trace_path.with_suffix(".html")
    out = build_dashboard(trace_path, output_path)
    print(f"[dashboard] wrote {out}")


if __name__ == "__main__":
    main()
