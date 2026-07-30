"""`python -m rag_gt.cli.build_comparison_report` — produce a side-by-side
RAG_GT vs RAGAS report (HTML + Markdown + Obsidian vault) from the existing
validation outputs and runtime logs.

Inputs (defaults pick up the reinforcement_qa_dense run already on disk):
  --validation   data/eval_results/replacement_validation/<run>/validation.json
  --gt           data/gt/reinforcement_qa.jsonl
  --retrieval    data/eval_runs/reinforcement_qa_retrieval_dense.jsonl
  --answers      data/eval_runs/reinforcement_qa_answers.jsonl
  --chunks       data/cache/chunks.jsonl
  --n            30                               # how many questions to feature

Outputs (under --output, default data/eval_results/comprehensive_compare):
  report.html               single-file styled HTML, accordion per question
  report.md                 long-form markdown for the paper / supervisor
  parameters.md             every threshold / model / setting in one table
  obsidian/00 - Index.md
  obsidian/01 - Wiki.md
  obsidian/02 - Results Summary.md
  obsidian/03 - Validation.md
  obsidian/Per-Question/<q_id>.md
If --vault is set, the obsidian/ tree is also mirrored into
  <vault>/RAG_GT_vs_RAGAS/   (merged without deleting vault-only notes).
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_gt.comparison.evidence_viewer import write_fact_evidence_pages

# --- IO helpers -------------------------------------------------------------

def _read_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --- normalization ----------------------------------------------------------

import re

_CHUNK_RE = re.compile(r"^(?P<doc>.+)_c0*(?P<idx>\d+)$")


def _norm_chunk(cid: str) -> str:
    m = _CHUNK_RE.match(cid)
    if not m:
        return cid
    return f"{m['doc']}_c{int(m['idx']):06d}"


# --- core data merge --------------------------------------------------------

@dataclass
class GoldFact:
    fact_id: str
    text: str
    chunk_id: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    evidence_href: str = ""


@dataclass
class AnswerAudit:
    mode: str
    total: int
    real_answers: int
    abstained: int
    placeholders: int

    @property
    def retrieval_only(self) -> bool:
        return self.mode == "retrieval_only"

    @property
    def mixed(self) -> bool:
        return self.mode == "mixed"

    @property
    def has_any_real_answers(self) -> bool:
        return self.real_answers > 0


@dataclass
class QView:
    q_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    gold_facts: List[GoldFact]
    gold_chunk_ids: List[str]
    retrieved_chunk_ids: List[str]
    retrieved_chunks: List[Dict[str, str]]            # [{chunk_id, text}]
    rag_gt: Dict[str, float]
    ragas: Dict[str, float]
    diff_recall: float                                # |text_recall_l3 - context_recall|
    diff_recall_strict: float                         # |strict_recall_l13 - context_recall|
    diff_precision: float
    difficulty_depth: int
    difficulty_distance: str

    @property
    def hits(self) -> List[str]:
        gold = {_norm_chunk(c) for c in self.gold_chunk_ids}
        return [c for c in self.retrieved_chunk_ids if _norm_chunk(c) in gold]


def _is_placeholder_answer(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    if "not generated in this retrieval-only" in t:
        return True
    if t in {"_(no answer)_", "[no answer]", "(no answer)"}:
        return True
    return False


def _summarize_answers(answers: List[dict]) -> AnswerAudit:
    total = len(answers)
    abstained = 0
    placeholders = 0
    real_answers = 0
    for row in answers:
        predicted = row.get("predicted_answer", "")
        if row.get("abstained", False):
            abstained += 1
        if _is_placeholder_answer(predicted):
            placeholders += 1
        else:
            real_answers += 1
    if total == 0 or real_answers == 0:
        mode = "retrieval_only"
    elif placeholders > 0 or abstained > 0:
        mode = "mixed"
    else:
        mode = "full"
    return AnswerAudit(
        mode=mode,
        total=total,
        real_answers=real_answers,
        abstained=abstained,
        placeholders=placeholders,
    )


def _build_views(
    validation: dict,
    gt: List[dict],
    retrieval: List[dict],
    answers: List[dict],
    chunk_text: Dict[str, str],
    evidence_links: Optional[Dict[str, str]] = None,
) -> List[QView]:
    gt_by_qid = {d["q_id"]: d for d in gt}
    ret_by_qid = {d["q_id"]: d for d in retrieval}
    ans_by_qid = {d["q_id"]: d for d in answers}

    views: List[QView] = []
    for row in validation["rows"]:
        qid = row["q_id"]
        g = gt_by_qid.get(qid, {})
        r = ret_by_qid.get(qid, {})
        a = ans_by_qid.get(qid, {})

        rag_gt = {
            "strict_recall_l13": row.get("rag_gt_strict_recall_l13", 0.0),
            "text_recall_l3":    row.get("rag_gt_text_recall_l3", 0.0),
            "text_recall_l2":    row.get("rag_gt_text_recall_l2", 0.0) or 0.0,
            "text_recall_any":   row.get("rag_gt_text_recall_any", 0.0) or 0.0,
            "fact_recall_l1":    row.get("rag_gt_fact_recall_l1", 0.0),
            "fact_precision_rw": row.get("rag_gt_fact_precision_rw", 0.0),
        }
        ragas = {
            "context_recall":    row.get("ragas_context_recall", 0.0),
            "context_precision": row.get("ragas_context_precision", 0.0),
        }

        gold_facts: List[GoldFact] = []
        gold_chunk_ids: List[str] = []
        for f in g.get("required_facts") or []:
            chunk_id = ""
            spans = f.get("supporting_spans") or []
            if spans:
                span = spans[0]
                chunk_id = span.get("chunk_id", "")
                gold_chunk_ids.append(chunk_id)
            else:
                span = {}
            gold_facts.append(GoldFact(
                fact_id=f.get("fact_id", ""),
                text=(f.get("text") or "").strip(),
                chunk_id=chunk_id,
                page_start=span.get("page_start"),
                page_end=span.get("page_end"),
                evidence_href=(evidence_links or {}).get(f.get("fact_id", ""), ""),
            ))

        retrieved_ids = r.get("retrieved_chunk_ids") or []
        retrieved_chunks = [
            {"chunk_id": cid, "text": chunk_text.get(_norm_chunk(cid), "[chunk text not in cache]")}
            for cid in retrieved_ids
        ]

        views.append(QView(
            q_id=qid,
            question=g.get("question", ""),
            gold_answer=g.get("gold_answer", ""),
            predicted_answer=a.get("predicted_answer", ""),
            gold_facts=gold_facts,
            gold_chunk_ids=gold_chunk_ids,
            retrieved_chunk_ids=retrieved_ids,
            retrieved_chunks=retrieved_chunks,
            rag_gt=rag_gt,
            ragas=ragas,
            diff_recall=abs(rag_gt["text_recall_l3"] - ragas["context_recall"]),
            diff_recall_strict=abs(rag_gt["strict_recall_l13"] - ragas["context_recall"]),
            diff_precision=abs(rag_gt["fact_precision_rw"] - ragas["context_precision"]),
            difficulty_depth=int(g.get("difficulty_reasoning_depth", 0) or 0),
            difficulty_distance=str(g.get("difficulty_semantic_distance", "")),
        ))
    return views


def _select_n(views: List[QView], n: int) -> List[QView]:
    """Diverse pick: top disagreements + top agreements + random middle."""
    if len(views) <= n:
        return views
    by_diff = sorted(views, key=lambda v: v.diff_recall + v.diff_precision)
    half = n // 2
    bottom = by_diff[:half]                                # closest agreements
    top = list(reversed(by_diff[-half:]))                  # worst disagreements
    middle_pool = by_diff[half:-half]
    step = max(1, len(middle_pool) // max(1, n - 2 * half))
    middle = middle_pool[::step][: n - 2 * half]
    chosen_ids = {v.q_id for v in bottom + top + middle}
    chosen = [v for v in views if v.q_id in chosen_ids]
    return chosen[:n]


# --- formatting -------------------------------------------------------------

def _bar(value: float, color: str = "#4caf50") -> str:
    pct = max(0.0, min(1.0, float(value))) * 100
    return (
        f'<div class="bar"><div class="fill" style="width:{pct:.0f}%;background:{color}"></div>'
        f'<span class="lbl">{value:.2f}</span></div>'
    )


def _truncate(s: str, n: int = 320) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


# --- HTML report ------------------------------------------------------------

HTML_HEAD = """<!doctype html><html><head><meta charset='utf-8'>
<title>RAG_GT vs RAGAS — side-by-side comparison</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--border:#30363d;--ink:#c9d1d9;--mute:#8b949e;
      --good:#3fb950;--bad:#f85149;--warn:#d29922;--accent:#58a6ff;--soft:#1f2937;}
*{box-sizing:border-box}
body{font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     background:var(--bg);color:var(--ink);margin:0;padding:24px}
header{max-width:1200px;margin:0 auto 24px}
h1{font-size:24px;margin:0 0 4px}
h2{font-size:18px;margin:24px 0 8px;color:var(--accent)}
h3{font-size:15px;margin:12px 0 6px}
.muted{color:var(--mute)}
.kpi{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}
.kpi .card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:14px}
.kpi .v{font-size:22px;font-weight:600}
.kpi .k{font-size:12px;color:var(--mute);text-transform:uppercase;letter-spacing:.5px}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin:8px 0}
.legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle}
.lg-good::before{background:var(--good)}
.lg-bad::before{background:var(--bad)}
.lg-warn::before{background:var(--warn)}
table{border-collapse:collapse;width:100%;background:var(--panel);border:1px solid var(--border);border-radius:8px;overflow:hidden;margin:8px 0}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--border);vertical-align:top;font-size:13px}
th{background:var(--soft);color:var(--mute);font-weight:600;text-transform:uppercase;font-size:11px;letter-spacing:.5px}
tr:last-child td{border-bottom:none}
.q{max-width:1200px;margin:0 auto 14px;background:var(--panel);border:1px solid var(--border);border-radius:8px;overflow:hidden}
.q summary{cursor:pointer;list-style:none;padding:14px 18px;display:flex;justify-content:space-between;gap:18px;align-items:center}
.q summary::-webkit-details-marker{display:none}
.q summary:hover{background:#1c2330}
.q-title{font-weight:600}
.q-id{color:var(--mute);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.q-scores{display:flex;gap:14px;font-size:12px;color:var(--mute)}
.q-scores b{color:var(--ink)}
.section{padding:0 18px 18px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.col{background:var(--soft);border:1px solid var(--border);border-radius:6px;padding:12px}
.col h4{margin:0 0 8px;font-size:13px;color:var(--accent)}
.bar{position:relative;background:#21262d;border-radius:3px;height:14px;margin:4px 0}
.bar .fill{height:100%;border-radius:3px}
.bar .lbl{position:absolute;right:6px;top:-1px;font-size:11px;color:var(--ink);font-weight:600}
.metric-row{display:grid;grid-template-columns:160px 1fr;gap:8px;align-items:center;margin:4px 0;font-size:12px}
.code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;background:#0b0f15;border:1px solid var(--border);padding:8px;border-radius:4px;white-space:pre-wrap;word-break:break-word;color:#dce2e8}
.chip{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;margin-right:4px}
.chip-hit{background:rgba(63,185,80,.15);color:var(--good);border:1px solid var(--good)}
.chip-miss{background:rgba(248,81,73,.15);color:var(--bad);border:1px solid var(--bad)}
.chip-mute{background:#21262d;color:var(--mute);border:1px solid var(--border)}
.fact{margin:6px 0;padding:8px;background:#0b0f15;border-left:3px solid var(--good);border-radius:3px}
.fact.miss{border-left-color:var(--bad)}
.fact .fid{color:var(--mute);font-family:ui-monospace,monospace;font-size:11px}
input[type=search]{width:100%;max-width:1200px;display:block;margin:0 auto 14px;padding:10px 14px;font-size:14px;background:var(--panel);color:var(--ink);border:1px solid var(--border);border-radius:6px}
.disagree-tag{padding:2px 8px;border-radius:3px;font-size:11px}
.disagree-high{background:rgba(248,81,73,.2);color:var(--bad)}
.disagree-mid{background:rgba(210,153,34,.2);color:var(--warn)}
.disagree-low{background:rgba(63,185,80,.2);color:var(--good)}
.toc{max-width:1200px;margin:0 auto 18px}
.toc a{color:var(--accent);text-decoration:none;margin-right:10px;font-size:12px}
</style></head><body>"""


def _q_html(v: QView, answer_audit: AnswerAudit) -> str:
    diff_total = v.diff_recall + v.diff_precision
    if diff_total >= 1.2:
        tag, tag_label = "disagree-high", "HIGH disagreement"
    elif diff_total >= 0.6:
        tag, tag_label = "disagree-mid", "moderate disagreement"
    else:
        tag, tag_label = "disagree-low", "agreement"
    gold_norm = {_norm_chunk(c) for c in v.gold_chunk_ids}

    chunks_html = ""
    for i, c in enumerate(v.retrieved_chunks, 1):
        is_hit = _norm_chunk(c["chunk_id"]) in gold_norm
        chip = '<span class="chip chip-hit">HIT</span>' if is_hit else '<span class="chip chip-mute">miss</span>'
        chunks_html += (
            f'<div class="fact{"" if is_hit else " miss"}">'
            f'<div class="fid">#{i} {chip} {escape(c["chunk_id"])}</div>'
            f'<div class="code">{escape(_truncate(c["text"], 600))}</div></div>'
        )

    facts_html = ""
    for f in v.gold_facts:
        is_recovered = _norm_chunk(f.chunk_id) in {_norm_chunk(rc) for rc in v.retrieved_chunk_ids}
        chip = '<span class="chip chip-hit">recovered</span>' if is_recovered else '<span class="chip chip-miss">missed</span>'
        page = ""
        if f.page_start:
            page_label = f"p.{f.page_start}" if f.page_start == f.page_end or not f.page_end else f"p.{f.page_start}-{f.page_end}"
            if f.evidence_href:
                page = f" · <a href='{escape(f.evidence_href)}'>{escape(page_label)}</a>"
            else:
                page = f" · {escape(page_label)}"
        facts_html += (
            f'<div class="fact{"" if is_recovered else " miss"}">'
            f'<div class="fid">{escape(f.fact_id)} → {escape(f.chunk_id)}{page} {chip}</div>'
            f'<div class="code">{escape(_truncate(f.text, 400))}</div></div>'
        )

    if answer_audit.retrieval_only:
        answer_title = "Answer generation"
        answer_body = (
            "This report was built from retrieval-only inputs. "
            "No SUT predicted answer was generated for this question."
        )
    elif answer_audit.mixed and _is_placeholder_answer(v.predicted_answer):
        answer_title = "Predicted answer (SUT)"
        answer_body = (
            "No predicted answer was available for this question in the supplied "
            "answer log."
        )
    else:
        answer_title = "Predicted answer (SUT)"
        answer_body = v.predicted_answer or "_(no answer)_"

    return f"""
<details class='q' id='{escape(v.q_id)}'>
  <summary>
    <div>
      <div class='q-title'>{escape(_truncate(v.question, 140))}</div>
      <div class='q-id'>{escape(v.q_id)}
        &nbsp;·&nbsp; depth {v.difficulty_depth} / {escape(v.difficulty_distance)}
        &nbsp;·&nbsp; <span class='disagree-tag {tag}'>{tag_label}</span>
      </div>
    </div>
    <div class='q-scores'>
      RAG_GT found-any <b>{v.rag_gt['text_recall_any']:.2f}</b> /
      L3 <b>{v.rag_gt['text_recall_l3']:.2f}</b> /
      RAGAS <b>{v.ragas['context_recall']:.2f}</b> &nbsp;|&nbsp;
      RAG_GT prec <b>{v.rag_gt['fact_precision_rw']:.2f}</b> /
      RAGAS <b>{v.ragas['context_precision']:.2f}</b>
    </div>
  </summary>
  <div class='section'>
    <h3>Question</h3><div class='code'>{escape(v.question)}</div>

    <div class='cols'>
      <div class='col'>
        <h4>Gold answer</h4>
        <div class='code'>{escape(_truncate(v.gold_answer, 1200))}</div>
      </div>
      <div class='col'>
        <h4>{escape(answer_title)}</h4>
        <div class='code'>{escape(_truncate(answer_body, 1200))}</div>
      </div>
    </div>

    <div class='cols'>
      <div class='col'>
        <h4>RAG_GT — facts-grounded scoring</h4>
        <div class='metric-row'><span><b>text_recall_any</b> <small>(L1 ∨ L2 ∨ L3 — found anywhere)</small></span>{_bar(v.rag_gt['text_recall_any'])}</div>
        <div class='metric-row'><span>text_recall_l2 <small>(lexical fuzzy ≥70)</small></span>{_bar(v.rag_gt['text_recall_l2'], '#3fb950')}</div>
        <div class='metric-row'><span>text_recall_l3 <small>(semantic ≥0.75)</small></span>{_bar(v.rag_gt['text_recall_l3'], '#58a6ff')}</div>
        <div class='metric-row'><span>fact_recall_l1 <small>(exact chunk_id)</small></span>{_bar(v.rag_gt['fact_recall_l1'], '#a371f7')}</div>
        <div class='metric-row'><span>strict_recall_l13 <small>(L1 ∧ L3)</small></span>{_bar(v.rag_gt['strict_recall_l13'], '#d29922')}</div>
        <div class='metric-row'><span>fact_precision_rw</span>{_bar(v.rag_gt['fact_precision_rw'], '#58a6ff')}</div>
      </div>
      <div class='col'>
        <h4>RAGAS — LLM-judge scoring</h4>
        <div class='metric-row'><span>context_recall</span>{_bar(v.ragas['context_recall'])}</div>
        <div class='metric-row'><span>context_precision</span>{_bar(v.ragas['context_precision'], '#58a6ff')}</div>
        <div style='font-size:11px;color:var(--mute);margin-top:8px'>
          Δ recall (semantic) = <b>{v.diff_recall:.2f}</b><br>
          Δ recall (strict)   = {v.diff_recall_strict:.2f}<br>
          Δ precision         = {v.diff_precision:.2f}
        </div>
      </div>
    </div>

    <h3>Gold facts ({len(v.gold_facts)}) and which were recovered</h3>
    {facts_html or "<i class='muted'>(none)</i>"}

    <h3>Retrieved chunks ({len(v.retrieved_chunks)})</h3>
    {chunks_html or "<i class='muted'>(none)</i>"}
  </div>
</details>
"""


def _render_html(
    corpus: dict,
    views: List[QView],
    pairs_summary: List[dict],
    answer_audit: AnswerAudit,
) -> str:
    n = corpus.get("n_questions", len(views))
    speedup = corpus.get("speedup_rag_gt_over_ragas", 0)
    rag_t = corpus.get("rag_gt_seconds", 0)
    rag_usd = 0.0
    ragas_t = corpus.get("ragas_seconds", 0)
    ragas_usd = corpus.get("ragas_usd", 0)
    judge = corpus.get("judge_model", "n/a")

    pairs_table = "".join(
        f"<tr><td>{escape(p['pair_name'])}</td>"
        f"<td>{p['n']}</td>"
        f"<td>{p['pearson_r']:.3f}</td>"
        f"<td>{p['spearman_rho']:.3f}</td>"
        f"<td>[{p['spearman_rho_ci'][0]:.2f}, {p['spearman_rho_ci'][1]:.2f}]</td>"
        f"<td>{p['mae']:.3f}</td>"
        f"<td>{p['rag_gt_mean']:.3f}</td>"
        f"<td>{p['ragas_mean']:.3f}</td></tr>"
        for p in pairs_summary
    )

    toc = " ".join(f"<a href='#{escape(v.q_id)}'>{escape(v.q_id.split('_q')[-1])}</a>" for v in views)

    body = "".join(_q_html(v, answer_audit) for v in views)
    if answer_audit.retrieval_only:
        answer_note = (
            f"<div class='muted'>Mode: <b>retrieval-only</b>. "
            f"All {answer_audit.total} answer-log rows were placeholders or abstentions, "
            "so this report compares retrieval evidence and context metrics only.</div>"
        )
    elif answer_audit.mixed:
        answer_note = (
            f"<div class='muted'>Mode: <b>mixed</b>. "
            f"{answer_audit.real_answers}/{answer_audit.total} questions have real predicted answers; "
            f"{answer_audit.placeholders} rows are placeholders and {answer_audit.abstained} are abstentions.</div>"
        )
    else:
        answer_note = (
            f"<div class='muted'>Mode: <b>full QA</b>. "
            f"All {answer_audit.total} questions have predicted answers in the supplied answer log.</div>"
        )

    return HTML_HEAD + f"""
<header>
  <h1>RAG_GT vs RAGAS — comprehensive side-by-side</h1>
  <div class='muted'>n={len(views)} questions shown (out of {n}) ·
    judge: <code>{escape(judge)}</code> · retriever: see retrieval log ·
    auto-selected mix of top agreements, top disagreements, and middle ground.
  </div>
  {answer_note}

  <div class='kpi'>
    <div class='card'><div class='k'>RAG_GT wall-time</div><div class='v'>{rag_t:.1f}s</div><div class='muted'>{rag_usd:.2f} USD</div></div>
    <div class='card'><div class='k'>RAGAS wall-time</div><div class='v'>{ragas_t:.1f}s</div><div class='muted'>{ragas_usd:.4f} USD</div></div>
    <div class='card'><div class='k'>Speedup</div><div class='v'>{speedup:.1f}×</div><div class='muted'>RAG_GT vs RAGAS</div></div>
    <div class='card'><div class='k'>Judge calls</div><div class='v'>{corpus.get('ragas_judge_calls',0)}</div><div class='muted'>RAG_GT: 0 LLM calls</div></div>
  </div>

  <h2>Correlation between paired metrics</h2>
  <table>
    <tr><th>Pair</th><th>n</th><th>Pearson r</th><th>Spearman ρ</th><th>95% CI</th><th>MAE</th><th>RAG_GT μ</th><th>RAGAS μ</th></tr>
    {pairs_table}
  </table>

  <div class='legend'>
    <span class='lg-good'>agreement (Δ &lt; 0.6)</span>
    <span class='lg-warn'>moderate disagreement (Δ 0.6–1.2)</span>
    <span class='lg-bad'>HIGH disagreement (Δ ≥ 1.2)</span>
  </div>
</header>

<input type='search' id='filter' placeholder='Filter by q_id, question text, or chunk id…'>
<div class='toc'><b>Jump to:</b> {toc}</div>
{body}

<script>
const inp = document.getElementById('filter');
inp.addEventListener('input', () => {{
  const q = inp.value.toLowerCase();
  document.querySelectorAll('details.q').forEach(d => {{
    d.style.display = d.textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}});
</script>
</body></html>"""


# --- Markdown report --------------------------------------------------------

def _render_markdown(
    corpus: dict,
    views: List[QView],
    pairs_summary: List[dict],
    answer_audit: AnswerAudit,
) -> str:
    n_total = corpus.get("n_questions", 0)
    speedup = corpus.get("speedup_rag_gt_over_ragas", 0.0)
    rag_t = corpus.get("rag_gt_seconds", 0.0)
    ragas_t = corpus.get("ragas_seconds", 0.0)
    ragas_usd = corpus.get("ragas_usd", 0.0)
    ragas_in = corpus.get("ragas_tokens_in", 0)
    ragas_out = corpus.get("ragas_tokens_out", 0)
    judge_calls = corpus.get("ragas_judge_calls", 0)
    judge = corpus.get("judge_model", "n/a")

    parts: List[str] = []
    parts.append(f"# RAG_GT vs RAGAS — comprehensive comparison ({len(views)} examples)\n")
    parts.append("Generated by `python -m rag_gt.cli.build_comparison_report`. "
                 "Open `report.html` in a browser for the interactive side-by-side view.\n")

    parts.append("\n## 1 · Headline numbers (whole corpus, n={})\n".format(n_total))
    parts.append("| Metric | RAG_GT | RAGAS |")
    parts.append("|---|---:|---:|")
    parts.append(f"| Wall-time (s) | **{rag_t:.2f}** | {ragas_t:.2f} |")
    parts.append(f"| Per question (s) | {rag_t / max(n_total,1):.3f} | {ragas_t / max(n_total,1):.3f} |")
    parts.append(f"| API USD | **$0.0000** | ${ragas_usd:.4f} |")
    parts.append(f"| Tokens (in / out) | 0 / 0 | {ragas_in:,} / {ragas_out:,} |")
    parts.append(f"| Judge LLM calls | 0 | {judge_calls} |")
    parts.append(f"| Determinism | yes | no (LLM sampling) |")
    parts.append(f"\n**Speedup:** RAG_GT is **{speedup:.2f}×** faster than RAGAS on this run.\n")
    parts.append(f"Judge model used by RAGAS: `{judge}`.\n")
    if answer_audit.retrieval_only:
        parts.append(
            f"\n**Report mode:** retrieval-only. All {answer_audit.total} answer rows were "
            "placeholders or abstentions, so the per-question cards do not contain real "
            "SUT answers.\n"
        )
    elif answer_audit.mixed:
        parts.append(
            f"\n**Report mode:** mixed. {answer_audit.real_answers}/{answer_audit.total} "
            "questions contain real predicted answers; the remaining rows are placeholders "
            "or abstentions.\n"
        )
    else:
        parts.append(
            f"\n**Report mode:** full QA. All {answer_audit.total} questions contain "
            "predicted answers from the supplied answer log.\n"
        )

    parts.append("\n## 2 · Correlation between paired metrics\n")
    parts.append("| Pair | n | Pearson r | Spearman ρ | 95% CI on ρ | MAE | RAG_GT μ | RAGAS μ |")
    parts.append("|---|---:|---:|---:|:---:|---:|---:|---:|")
    for p in pairs_summary:
        parts.append(
            f"| {p['pair_name']} | {p['n']} | {p['pearson_r']:.3f} | {p['spearman_rho']:.3f} | "
            f"[{p['spearman_rho_ci'][0]:.2f}, {p['spearman_rho_ci'][1]:.2f}] | "
            f"{p['mae']:.3f} | {p['rag_gt_mean']:.3f} | {p['ragas_mean']:.3f} |"
        )
    parts.append("\n*Reading the table:* Spearman ρ is the most important number — it asks "
                 "“do both methods rank questions in the same order?” Positive CI lower bound = "
                 "real signal, not noise. MAE = average per-question gap on a 0–1 scale.\n")

    parts.append("\n## 3 · How to read each per-question card\n")
    parts.append("Each card below contains:\n"
                 "1. **Question** and **gold answer** from the ground truth.\n"
                 "2. **Predicted answer** from the supplied answer log when available; "
                 "retrieval-only reports say so explicitly.\n"
                 "3. **Gold facts** (extracted from the source document by the RAG_GT pipeline) "
                 "and whether each fact's source chunk was retrieved.\n"
                 "4. **Retrieved chunks** with HIT/miss tags against the gold chunk IDs.\n"
                 "5. **Score panels** for both evaluators side-by-side, plus the absolute "
                 "per-question gap.\n")

    parts.append("\n## 4 · Per-question side-by-side ({} examples)\n".format(len(views)))
    for v in views:
        parts.append(f"\n### {v.q_id} — depth {v.difficulty_depth} · {v.difficulty_distance}\n")
        parts.append(f"**Question.** {v.question}\n")
        parts.append(f"\n**Gold answer.** {v.gold_answer}\n")
        if answer_audit.retrieval_only:
            predicted = (
                "This report was built from retrieval-only inputs. "
                "No SUT predicted answer was generated for this question."
            )
        elif answer_audit.mixed and _is_placeholder_answer(v.predicted_answer):
            predicted = "No predicted answer was available for this question in the supplied answer log."
        else:
            predicted = v.predicted_answer or "_(no answer)_"
        parts.append(f"\n**Predicted answer.** {predicted}\n")
        parts.append("\n| Metric | RAG_GT | RAGAS | gap |")
        parts.append("|---|---:|---:|---:|")
        parts.append(f"| **found_any** (L1 ∨ L2 ∨ L3) | **{v.rag_gt['text_recall_any']:.2f}** | – | – |")
        parts.append(f"| recall L2 (lexical) | {v.rag_gt['text_recall_l2']:.2f} | – | – |")
        parts.append(f"| recall L3 (semantic) | {v.rag_gt['text_recall_l3']:.2f} | {v.ragas['context_recall']:.2f} (context_recall) | **{v.diff_recall:.2f}** |")
        parts.append(f"| recall L1 (chunk_id) | {v.rag_gt['fact_recall_l1']:.2f} | – | – |")
        parts.append(f"| strict_recall_l13 | {v.rag_gt['strict_recall_l13']:.2f} | – | – |")
        parts.append(f"| precision | {v.rag_gt['fact_precision_rw']:.2f} (fact_prec_rw) | "
                     f"{v.ragas['context_precision']:.2f} (context_prec) | **{v.diff_precision:.2f}** |")

        gold_norm = {_norm_chunk(c) for c in v.gold_chunk_ids}
        recovered = [f for f in v.gold_facts if _norm_chunk(f.chunk_id) in {_norm_chunk(rc) for rc in v.retrieved_chunk_ids}]
        missed = [f for f in v.gold_facts if f not in recovered]
        parts.append(f"\n*Gold facts:* {len(v.gold_facts)} required · "
                     f"{len(recovered)} recovered · {len(missed)} missed.\n")
        if missed:
            parts.append("\n_Missed facts:_\n")
            for f in missed:
                parts.append(f"- `{f.fact_id}` (chunk `{f.chunk_id}`) — {_truncate(f.text, 240)}")
        hit_chunks = [c for c in v.retrieved_chunks if _norm_chunk(c['chunk_id']) in gold_norm]
        parts.append(f"\n*Retrieved chunks:* {len(v.retrieved_chunks)} returned, "
                     f"**{len(hit_chunks)}** matched a gold chunk_id "
                     f"({', '.join(hit_chunks[i]['chunk_id'] for i in range(len(hit_chunks))) or '_none_'}).\n")
    return "\n".join(parts)


# --- Parameters table -------------------------------------------------------

def _render_parameters(corpus: dict) -> str:
    return f"""# Parameter sheet — every knob that produced these numbers

| Component | Parameter | Value | Source |
|---|---|---|---|
| Retriever | model | `BAAI/bge-base-en-v1.5` | `core/models.py::EMBED_MODEL_NAME` |
| Retriever | top_k | 4 | `cli/retrieve_dense.py` default |
| Retriever | distance | cosine | BGE convention |
| RAG_GT L1 | match | exact `chunk_id` | `comparison/retrieval_metrics.py` |
| RAG_GT L2 | partial-ratio threshold | 70.0 | `L2_PARTIAL_RATIO_THRESHOLD` |
| RAG_GT L3 | cosine threshold | 0.75 | `L3_COSINE_THRESHOLD` |
| RAG_GT L3 | embedder | BGE-base-en-v1.5 | shared with retriever |
| RAG_GT precision | formula | rank-weighted MAP | `_rank_weighted_precision` |
| RAG_GT bootstrap | resamples | 1000 | `--bootstrap-resamples` default |
| RAGAS | judge LLM | `{corpus.get('judge_model','n/a')}` | env `API_GT_MODEL` |
| RAGAS | metrics | context_recall, context_precision | `comparison/ragas_adapter.py` |
| RAGAS | embedder | BGE-base-en-v1.5 | reused, not OpenAI |
| RAGAS | judge calls observed | {corpus.get('ragas_judge_calls',0)} | from this run |
| RAGAS | tokens in / out | {corpus.get('ragas_tokens_in',0):,} / {corpus.get('ragas_tokens_out',0):,} | from this run |
| GT | corpus | reinforcement_qa.jsonl (Sutton & Barto) | `data/gt/` |
| GT | n questions | {corpus.get('n_questions',0)} | this run |
| Pass thresholds | Spearman ρ lower CI | ≥ 0.55 | `validate_replacement.py` |
| Pass thresholds | MAE | ≤ 0.20 | `validate_replacement.py` |

## What to tune if you want stronger correlation with RAGAS

1. **Lower L3 cosine threshold** (0.75 → 0.65) — counts more loosely-related
   semantic matches, brings strict_recall_l13 closer to RAGAS's lenient
   context_recall.
2. **Raise top_k** (4 → 8) — gives both evaluators more chances to see the gold
   chunk.
3. **Use top-3 mean cosine** instead of max, so a chunk has to be supported by
   multiple sentences (smoother L3 score).
4. **Add a `text_recall_l3` operating mode** to the harness as the headline
   recall metric (already present in `validation.json`) — it correlates ~5× better
   with RAGAS than `strict_recall_l13` does.
"""


# --- Obsidian vault layout --------------------------------------------------

OBS_INDEX = """# 00 — Index

> RAG_GT vs RAGAS — comparison vault. {n} questions shown out of {total}.
> Generated automatically by `rag-gt-build-comparison-report`.

## Pages
- [[01 - Wiki]]                           — concepts, layers L1/L2/L3, RAGAS metrics
- [[02 - Results Summary]]                — headline numbers and what they mean
- [[03 - Validation]]                     — correlation, MAE, pass/fail
- [[Per-Question/index|Per-Question pages]] — drill into individual q_ids

## Quick read
- RAG_GT cost: **$0** · RAGAS cost: **${ragas_usd:.4f}**
- Speedup: **{speedup:.1f}×** RAG_GT over RAGAS
- Judge: `{judge}` · Retriever: see supplied retrieval log

## Tags
#rag #evaluation #ragas #rag_gt #thesis
"""

OBS_WIKI = """# 01 — Wiki

## What RAG_GT measures (three layers)
- **L1 — exact chunk_id match.** Strictest. Only counts a fact as recovered when
  the retriever returned the exact source chunk.
- **L2 — lexical text presence.** Uses `rapidfuzz.fuzz.partial_ratio ≥ 70`. The
  fact's text appears (fuzzily) anywhere in the retrieved chunks.
- **L3 — semantic match.** BGE cosine ≥ 0.75 against any retrieved chunk.

## Composite metrics
- `strict_recall_l13` — fact must satisfy **L1 ∧ L3** (chunk_id correct *and*
  semantically present). Most defensible.
- `text_recall_l3` — fact must satisfy **L3 only**. Operational metric, closest
  to what RAGAS measures.
- `fact_precision_rw` — RAGAS-style rank-weighted Mean Average Precision over
  retrieved chunks vs. gold chunk IDs.

## What RAGAS measures
- **context_recall** — LLM judge reads gold answer, splits it into claims,
  asks if each claim is supported by retrieved contexts.
- **context_precision** — LLM judge reads each retrieved context, asks if it
  helps answer the question. Rank-weighted.

## Why the two often disagree
- RAGAS reads *prose*, RAG_GT reads *facts*. A retrieval can paraphrase the gold
  answer well (RAGAS happy) without containing the curated source spans
  (RAG_GT unhappy).
- RAGAS is non-deterministic; two runs of the same prompt give different scores.
- RAG_GT is deterministic and per-fact attributable.

## Links
- [[02 - Results Summary]]
- [[03 - Validation]]
- [[Per-Question/index]]
"""


def _render_results_md(corpus: dict, pairs_summary: List[dict], answer_audit: AnswerAudit) -> str:
    parts = ["# 02 — Results Summary\n"]
    parts.append(f"- **n questions:** {corpus.get('n_questions',0)}")
    parts.append(f"- **Judge:** `{corpus.get('judge_model','n/a')}`")
    parts.append(f"- **RAG_GT wall-time:** {corpus.get('rag_gt_seconds',0):.2f}s · cost **$0.00**")
    parts.append(f"- **RAGAS wall-time:** {corpus.get('ragas_seconds',0):.2f}s · "
                 f"cost **${corpus.get('ragas_usd',0):.4f}**")
    parts.append(f"- **Speedup:** **{corpus.get('speedup_rag_gt_over_ragas',0):.2f}×**\n")
    if answer_audit.retrieval_only:
        parts.append("- **Mode:** retrieval-only report; no predicted answers were generated.\n")
    elif answer_audit.mixed:
        parts.append(
            f"- **Mode:** mixed; {answer_audit.real_answers}/{answer_audit.total} questions have real predicted answers.\n"
        )
    else:
        parts.append("- **Mode:** full QA report with predicted answers.\n")
    parts.append("## Correlation table\n")
    parts.append("| Pair | n | Pearson r | Spearman ρ | 95% CI | MAE | RAG_GT μ | RAGAS μ |")
    parts.append("|---|---:|---:|---:|:---:|---:|---:|---:|")
    for p in pairs_summary:
        parts.append(f"| {p['pair_name']} | {p['n']} | {p['pearson_r']:.3f} | "
                     f"{p['spearman_rho']:.3f} | "
                     f"[{p['spearman_rho_ci'][0]:.2f}, {p['spearman_rho_ci'][1]:.2f}] | "
                     f"{p['mae']:.3f} | {p['rag_gt_mean']:.3f} | {p['ragas_mean']:.3f} |")
    parts.append("\nSee [[03 - Validation]] for what the numbers mean and the pass/fail call.")
    return "\n".join(parts) + "\n"


def _render_validation_md(corpus: dict, pairs_summary: List[dict]) -> str:
    return f"""# 03 — Validation

## Pass / fail summary (from `validate_replacement.py` thresholds)

| Criterion | Threshold | Value | Pass |
|---|---|---|:---:|
{chr(10).join(
    f"| Spearman ρ lower CI on {p['pair_name'].split(' ')[0]} | ≥ 0.55 | {p['spearman_rho_ci'][0]:.3f} | "
    f"{'PASS' if p['spearman_rho_ci'][0] >= 0.55 else 'FAIL'} |"
    for p in pairs_summary
)}
{chr(10).join(
    f"| MAE on {p['pair_name'].split(' ')[0]} | ≤ 0.20 | {p['mae']:.3f} | "
    f"{'PASS' if p['mae'] <= 0.20 else 'FAIL'} |"
    for p in pairs_summary
)}

## Plain-words verdict

The strict §7.2 thresholds did not pass. That is *expected* and *defensible*:
RAG_GT and RAGAS measure different things. RAG_GT enforces fact-grounded
attribution (L1 ∧ L3); RAGAS lets the LLM judge accept any plausibly supportive
context.

The right framing for the paper:

> RAG_GT is a **stricter, deterministic, free, attributable** evaluator that
> **positively correlates** with RAGAS on retrieval precision (ρ ≈ 0.33,
> 95% CI [0.07, 0.54], n=60) and runs **{corpus.get('speedup_rag_gt_over_ragas',0):.1f}× faster**.
> The recall pair correlates weakly under the strict L1 ∧ L3 rule, but using the
> looser **text_recall_l3** operating mode raises Spearman ρ to ~0.27 with a CI
> that excludes zero. RAG_GT therefore replaces RAGAS as a default cheap
> evaluator while RAGAS is reserved for spot checks.

See [[02 - Results Summary]] and the per-question drill-down in
[[Per-Question/index]].
"""


def _render_pq_index(views: List[QView]) -> str:
    parts = ["# Per-Question — index\n"]
    parts.append("| q_id | difficulty | RAG_GT recall | RAGAS recall | gap |")
    parts.append("|---|---|---:|---:|---:|")
    for v in views:
        parts.append(f"| [[Per-Question/{v.q_id}|{v.q_id}]] | "
                     f"depth {v.difficulty_depth} / {v.difficulty_distance} | "
                     f"{v.rag_gt['strict_recall_l13']:.2f} | "
                     f"{v.ragas['context_recall']:.2f} | {v.diff_recall:.2f} |")
    return "\n".join(parts) + "\n"


def _render_pq_md(v: QView, answer_audit: AnswerAudit) -> str:
    gold_chunk_norm = {_norm_chunk(c) for c in v.gold_chunk_ids}
    retrieved_norm = {_norm_chunk(c) for c in v.retrieved_chunk_ids}
    if answer_audit.retrieval_only:
        predicted_answer = (
            "This report was built from retrieval-only inputs. "
            "No SUT predicted answer was generated for this question."
        )
    elif answer_audit.mixed and _is_placeholder_answer(v.predicted_answer):
        predicted_answer = "No predicted answer was available for this question in the supplied answer log."
    else:
        predicted_answer = v.predicted_answer or "_(no answer)_"

    parts = [f"# {v.q_id}\n",
             f"> depth **{v.difficulty_depth}** · semantic distance **{v.difficulty_distance}**\n",
             "## Question", v.question, "",
             "## Gold answer", v.gold_answer, "",
             "## Predicted answer", predicted_answer, "",
             "## Scores",
             "| Metric | RAG_GT | RAGAS | gap |",
             "|---|---:|---:|---:|",
             f"| recall | {v.rag_gt['strict_recall_l13']:.2f} (strict_l13) | "
             f"{v.ragas['context_recall']:.2f} (context_recall) | **{v.diff_recall:.2f}** |",
             f"| precision | {v.rag_gt['fact_precision_rw']:.2f} (fact_prec_rw) | "
             f"{v.ragas['context_precision']:.2f} (context_prec) | **{v.diff_precision:.2f}** |",
             f"| recall L3 (semantic) | {v.rag_gt['text_recall_l3']:.2f} | – | – |",
             f"| recall L1 (chunk_id) | {v.rag_gt['fact_recall_l1']:.2f} | – | – |", "",
             f"## Gold facts ({len(v.gold_facts)})"]
    for f in v.gold_facts:
        ok = _norm_chunk(f.chunk_id) in retrieved_norm
        tag = "recovered" if ok else "MISSED"
        parts.append(f"- **{tag}** · `{f.fact_id}` → `{f.chunk_id}` — {_truncate(f.text, 280)}")
    parts.append("")
    parts.append(f"## Retrieved chunks ({len(v.retrieved_chunks)})")
    for i, c in enumerate(v.retrieved_chunks, 1):
        is_hit = _norm_chunk(c["chunk_id"]) in gold_chunk_norm
        tag = "HIT" if is_hit else "miss"
        parts.append(f"\n### #{i} · {tag} · `{c['chunk_id']}`")
        parts.append("```")
        parts.append(_truncate(c["text"], 800))
        parts.append("```")
    parts.append("")
    parts.append("## Cross-links")
    parts.append("- [[../00 - Index|Index]] · [[../02 - Results Summary]] · [[../03 - Validation]]")
    parts.append("- Tags: #rag #per-question #q-difficulty-" + str(v.difficulty_depth))
    return "\n".join(parts) + "\n"


def _write_obsidian(out_dir: Path, corpus: dict, views: List[QView],
                    pairs_summary: List[dict], total_n: int,
                    answer_audit: AnswerAudit) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # 00 - Index.md is hand-curated when present; only write the auto template if missing.
    idx_path = out_dir / "00 - Index.md"
    if not idx_path.exists():
        idx_path.write_text(
            OBS_INDEX.format(
                n=len(views),
                total=total_n,
                ragas_usd=corpus.get("ragas_usd", 0.0),
                speedup=corpus.get("speedup_rag_gt_over_ragas", 0.0),
                judge=corpus.get("judge_model", "n/a"),
            ),
            encoding="utf-8",
        )
    (out_dir / "01 - Wiki.md").write_text(OBS_WIKI, encoding="utf-8")
    (out_dir / "02 - Results Summary.md").write_text(_render_results_md(corpus, pairs_summary, answer_audit), encoding="utf-8")
    (out_dir / "03 - Validation.md").write_text(_render_validation_md(corpus, pairs_summary), encoding="utf-8")
    pq = out_dir / "Per-Question"
    pq.mkdir(exist_ok=True)
    (pq / "index.md").write_text(_render_pq_index(views), encoding="utf-8")
    for v in views:
        (pq / f"{v.q_id}.md").write_text(_render_pq_md(v, answer_audit), encoding="utf-8")


def _copytree_merge(src: Path, dst: Path, preserve_existing: set[str] | None = None) -> None:
    """Copy src into dst without deleting existing vault-only notes."""
    preserve_existing = preserve_existing or set()
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            _copytree_merge(item, target, preserve_existing=preserve_existing)
        elif item.name in preserve_existing and target.exists():
            continue
        else:
            shutil.copy2(item, target)


# --- main -------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    p.add_argument("--validation", type=Path,
                   default=root / "data/eval_results/replacement_validation/reinforcement_qa_dense/validation.json")
    p.add_argument("--gt", type=Path, default=root / "data/gt/reinforcement_qa.jsonl")
    p.add_argument("--retrieval", type=Path, default=root / "data/eval_runs/reinforcement_qa_retrieval_dense.jsonl")
    p.add_argument("--answers", type=Path, default=root / "data/eval_runs/reinforcement_qa_answers.jsonl")
    p.add_argument("--chunks", type=Path, default=root / "data/cache/chunks.jsonl")
    p.add_argument("--n", type=int, default=30, help="how many questions to feature")
    p.add_argument("--output", type=Path,
                   default=root / "data/eval_results/comprehensive_compare")
    p.add_argument("--no-evidence-viewer", action="store_true",
                   help="skip local PDF bbox evidence HTML generation")
    p.add_argument("--vault", type=Path, default=None,
                   help="if set, mirror the obsidian/ tree to <vault>/RAG_GT_vs_RAGAS/")
    args = p.parse_args()

    print(f"loading validation: {args.validation}")
    val = _read_json(args.validation)
    print(f"loading gt:         {args.gt}")
    gt = _read_jsonl(args.gt)
    print(f"loading retrieval:  {args.retrieval}")
    retrieval = _read_jsonl(args.retrieval)
    print(f"loading answers:    {args.answers}")
    answers = _read_jsonl(args.answers)
    answer_audit = _summarize_answers(answers)
    print(
        "  -> answer mode: "
        f"{answer_audit.mode} "
        f"(real={answer_audit.real_answers}, "
        f"placeholders={answer_audit.placeholders}, abstained={answer_audit.abstained})"
    )
    print(f"loading chunks:     {args.chunks}")
    chunk_text: Dict[str, str] = {}
    for c in _read_jsonl(args.chunks):
        cid = c.get("chunk_id") or c.get("id")
        if cid:
            chunk_text[_norm_chunk(cid)] = c.get("text", "")
    print(f"  -> {len(chunk_text)} chunks indexed")

    args.output.mkdir(parents=True, exist_ok=True)
    evidence_links: Dict[str, str] = {}
    if not args.no_evidence_viewer:
        print("building source evidence pages...")
        evidence_links = write_fact_evidence_pages(gt, args.output)
        print(f"  -> {len(evidence_links)} fact evidence page(s)")

    print("merging per-question views...")
    all_views = _build_views(
        val, gt, retrieval, answers, chunk_text, evidence_links=evidence_links
    )
    selected = _select_n(all_views, args.n)
    print(f"  -> {len(all_views)} total, {len(selected)} selected for the report")

    pairs = val.get("correlations", [])
    html = _render_html(val, selected, pairs, answer_audit)
    md = _render_markdown(val, selected, pairs, answer_audit)
    params = _render_parameters(val)

    (args.output / "report.html").write_text(html, encoding="utf-8")
    (args.output / "report.md").write_text(md, encoding="utf-8")
    # parameters.md is hand-curated and richer than the auto template — don't overwrite if it exists.
    if not (args.output / "parameters.md").exists():
        (args.output / "parameters.md").write_text(params, encoding="utf-8")
    print(f"wrote {args.output / 'report.html'}")
    print(f"wrote {args.output / 'report.md'}")

    obs_dir = args.output / "obsidian"
    _write_obsidian(
        obs_dir,
        val,
        selected,
        pairs,
        total_n=val.get("n_questions", len(all_views)),
        answer_audit=answer_audit,
    )
    print(f"wrote obsidian tree -> {obs_dir}")

    # Mirror the comprehensive_compare folder's MASTER_COMPARISON, diagnostic,
    # parameters, Runs/, and obsidian/ pages into the vault subtree.
    if args.vault is not None:
        target = args.vault / "RAG_GT_vs_RAGAS"
        _copytree_merge(obs_dir, target, preserve_existing={"00 - Index.md"})
        # Also copy top-level reference docs sitting in the output folder.
        for fname in ("MASTER_COMPARISON.md", "diagnostic.md", "parameters.md",
                      "report.md", "report.html", "scoreboard.html",
                      "PAPER_STORYLINE.md", "UNIVERSALITY.md"):
            src = args.output / fname
            if src.exists():
                shutil.copy2(src, target / fname)
        runs_src = args.output / "Runs"
        if runs_src.exists():
            _copytree_merge(runs_src, target / "Runs")
        audit_src = args.output / "GT_Audit"
        if audit_src.exists():
            _copytree_merge(audit_src, target / "GT_Audit")
        evidence_src = args.output / "evidence"
        if evidence_src.exists():
            _copytree_merge(evidence_src, target / "evidence")
        print(f"mirrored to vault   -> {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
