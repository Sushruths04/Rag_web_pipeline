"""`python -m rag_gt.cli.build_master_report` — build the master comparison
markdown table that summarizes every retrieval config we have run, plus an
HTML scoreboard, and (optionally) mirror the whole bundle to an Obsidian vault.

Reads:
  data/eval_results/comprehensive_compare/all_configs.json
  data/eval_results/replacement_validation/<run>/validation.json   (multiple)

Writes:
  data/eval_results/comprehensive_compare/MASTER_COMPARISON.md   (rewritten)
  data/eval_results/comprehensive_compare/scoreboard.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _safe(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _table_row(name: str, retr: str, top_k: int, n: int,
               rg_l3: float, rg_l1: float, rg_strict: float, rg_prec: float,
               ragas_recall: Optional[float], ragas_prec: Optional[float],
               rho_recall: Optional[float], rho_prec: Optional[float],
               mae_recall: Optional[float], mae_prec: Optional[float],
               rag_gt_s: float, ragas_s: Optional[float], ragas_usd: Optional[float],
               file_link: str) -> str:
    def fmt(v, places=3):
        return "—" if v is None else (f"{v:.{places}f}" if isinstance(v, float) else str(v))
    return ("| " + " | ".join([
        name, retr, str(top_k), str(n),
        fmt(rg_l3), fmt(rg_l1), fmt(rg_strict), fmt(rg_prec),
        fmt(ragas_recall), fmt(ragas_prec),
        fmt(rho_recall), fmt(rho_prec),
        fmt(mae_recall), fmt(mae_prec),
        fmt(rag_gt_s, 1), fmt(ragas_s, 1), fmt(ragas_usd, 4),
        file_link,
    ]) + " |")


HTML_HEAD = """<!doctype html><html><head><meta charset='utf-8'><title>Scoreboard — RAG_GT vs RAGAS by retriever config</title>
<style>
body{font:14px -apple-system,sans-serif;background:#0d1117;color:#c9d1d9;padding:24px}
h1{font-size:22px}
table{border-collapse:collapse;width:100%;background:#161b22;border:1px solid #30363d;border-radius:8px;margin-top:14px}
th,td{padding:8px 10px;border-bottom:1px solid #30363d;text-align:right;font-size:13px}
th{background:#1f2937;color:#8b949e;text-transform:uppercase;font-size:11px;letter-spacing:.5px;text-align:right}
th.l,td.l{text-align:left}
.best{background:rgba(63,185,80,.15);font-weight:600}
.worst{background:rgba(248,81,73,.10)}
.muted{color:#8b949e;font-size:12px}
</style></head><body>"""


def _render_html(rows: List[dict]) -> str:
    headers = ["Run id","Retriever","top_k","n",
               "text_recall_l3","fact_recall (L1)","strict_l13","fact_prec_rw",
               "RAGAS ctx_recall","RAGAS ctx_prec",
               "ρ recall","ρ prec","MAE recall","MAE prec",
               "RAG_GT s","RAGAS s","RAGAS $"]
    # find best for highlighting
    def best_idx(key, hi=True):
        vals = [(i, r.get(key)) for i, r in enumerate(rows) if r.get(key) is not None]
        if not vals: return -1
        return max(vals, key=lambda t: t[1])[0] if hi else min(vals, key=lambda t: t[1])[0]
    best_l3 = best_idx("text_recall_l3")
    best_l1 = best_idx("fact_recall")
    best_prec = best_idx("fact_precision_rw")

    body = []
    body.append("<table><tr>" + "".join(
        f"<th class='l'>{h}</th>" if i < 2 else f"<th>{h}</th>"
        for i, h in enumerate(headers)
    ) + "</tr>")
    for i, r in enumerate(rows):
        cells = [
            ("l", r.get("name","")),
            ("l", r.get("retriever","")),
            ("", r.get("top_k","")),
            ("", r.get("n","")),
            ("best" if i == best_l3 else "", f"{r.get('text_recall_l3',0):.3f}"),
            ("best" if i == best_l1 else "", f"{r.get('fact_recall',0):.3f}"),
            ("", f"{r.get('strict_recall_l13',0):.3f}"),
            ("best" if i == best_prec else "", f"{r.get('fact_precision_rw',0):.3f}"),
            ("", "—" if r.get("ragas_context_recall") is None else f"{r['ragas_context_recall']:.3f}"),
            ("", "—" if r.get("ragas_context_precision") is None else f"{r['ragas_context_precision']:.3f}"),
            ("", "—" if r.get("rho_recall") is None else f"{r['rho_recall']:.3f}"),
            ("", "—" if r.get("rho_precision") is None else f"{r['rho_precision']:.3f}"),
            ("", "—" if r.get("mae_recall") is None else f"{r['mae_recall']:.3f}"),
            ("", "—" if r.get("mae_precision") is None else f"{r['mae_precision']:.3f}"),
            ("", f"{r.get('rag_gt_seconds',0):.1f}"),
            ("", "—" if r.get("ragas_seconds") is None else f"{r['ragas_seconds']:.1f}"),
            ("", "—" if r.get("ragas_usd") is None else f"${r['ragas_usd']:.4f}"),
        ]
        body.append("<tr>" + "".join(
            f"<td class='{cls}'>{val}</td>" if cls in ("l","best","worst")
            else f"<td>{val}</td>"
            for cls, val in cells
        ) + "</tr>")
    body.append("</table>")
    body.append("<p class='muted'>Best in each column highlighted in green. <b>—</b> means RAGAS was not re-run for that config.</p>")
    return HTML_HEAD + f"<h1>RAG_GT vs RAGAS — scoreboard across retriever configs</h1>\n" + "\n".join(body) + "\n</body></html>"


def main() -> int:
    p = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[3]
    p.add_argument("--all-configs", type=Path,
                   default=root / "data/eval_results/comprehensive_compare/all_configs.json")
    p.add_argument("--validation-bge", type=Path,
                   default=root / "data/eval_results/replacement_validation/reinforcement_qa_dense/validation.json")
    p.add_argument("--validation-hybrid-rerank", type=Path,
                   default=root / "data/eval_results/replacement_validation/hybrid_rerank/validation.json")
    p.add_argument("--din-configs", type=Path,
                   default=root / "data/eval_results/comprehensive_compare/din_configs.json")
    p.add_argument("--output", type=Path,
                   default=root / "data/eval_results/comprehensive_compare/scoreboard.html")
    args = p.parse_args()

    all_configs = _load_json(args.all_configs)["configs"]
    if args.din_configs.exists():
        all_configs = {**all_configs, **_load_json(args.din_configs)["configs"]}

    # Per-config table seed.
    base = [
        {"name": "BGE-d-5",     "retriever": "BGE dense",                         "top_k": 5,
         "key": "BGE-d-5",      "ragas_run": args.validation_bge,
         "file": "2026-05-12_BGE_topk5"},
        {"name": "BGE-d-20",    "retriever": "BGE dense",                         "top_k": 20,
         "key": "BGE-d-20",     "ragas_run": None,
         "file": "2026-05-12_BGE_topk20"},
        {"name": "Hyb-5",       "retriever": "Hybrid BM25+BGE",                   "top_k": 5,
         "key": "Hyb-5",        "ragas_run": None,
         "file": "2026-05-12_Hybrid_topk5"},
        {"name": "Hyb+RR-5",    "retriever": "Hybrid + bge-reranker",             "top_k": 5,
         "key": "Hyb+RR-5",     "ragas_run": args.validation_hybrid_rerank,
         "file": "2026-05-12_Hybrid_rerank_topk5"},
        {"name": "E5+RR-5",     "retriever": "Hybrid e5-large + bge-reranker",    "top_k": 5,
         "key": "E5+RR-5",      "ragas_run": None,
         "file": "2026-05-12_E5_hybrid_rerank_topk5"},
        {"name": "QR+Hyb+RR-5", "retriever": "QR + Hybrid + bge-reranker",        "top_k": 5,
         "key": "QR+Hyb+RR-5",  "ragas_run": None,
         "file": "2026-05-12_QR_hybrid_rerank_topk5"},
        {"name": "E5+QR+RR-5",  "retriever": "QR + Hybrid e5-large + bge-reranker", "top_k": 5,
         "key": "E5+QR+RR-5",   "ragas_run": None,
         "file": "2026-05-12_E5_QR_hybrid_rerank_topk5"},
        {"name": "FactAnc+RR-5","retriever": "Fact-anchored chunks + Hybrid + reranker", "top_k": 5,
         "key": "FactAnc+RR-5", "ragas_run": None,
         "file": "2026-05-12_FactAnchored_rerank_topk5"},
        {"name": "FactIdx-5",   "retriever": "Fact-indexed (RAG_GT-native)", "top_k": 5,
         "key": "FactIdx-5",
         "ragas_run": root / "data/eval_results/replacement_validation/fact_indexed/validation.json",
         "file": "2026-05-12_FactIdx_topk5"},
        {"name": "DIN-BGE-d-5",  "retriever": "DIN: BGE dense", "top_k": 5,
         "key": "DIN-BGE-d-5", "source": "din",
         "ragas_run": None, "file": "2026-05-12_DIN_universality"},
        {"name": "DIN-Hyb+RR-5", "retriever": "DIN: Hybrid + bge-reranker", "top_k": 5,
         "key": "DIN-Hyb+RR-5", "source": "din",
         "ragas_run": root / "data/eval_results/replacement_validation/din_hybrid_rerank/validation.json",
         "file": "2026-05-12_DIN_universality"},
        {"name": "DIN-FactIdx-5", "retriever": "DIN: Fact-indexed", "top_k": 5,
         "key": "DIN-FactIdx-5", "source": "din",
         "ragas_run": root / "data/eval_results/replacement_validation/din_fact_indexed/validation.json",
         "file": "2026-05-12_DIN_universality"},
    ]

    rows: List[dict] = []
    for r in base:
        cfg = all_configs.get(r["key"], {})
        means = cfg.get("means", {})
        out = {
            "name": r["name"],
            "retriever": r["retriever"],
            "top_k": r["top_k"],
            "n": 60,
            "file": r.get("file", ""),
            "text_recall_l3": means.get("text_recall_l3"),
            "fact_recall": means.get("fact_recall"),
            "strict_recall_l13": means.get("strict_recall_l13"),
            "fact_precision_rw": means.get("fact_precision_rw"),
            "rag_gt_seconds": cfg.get("rag_gt_seconds"),
        }
        if r["ragas_run"] and r["ragas_run"].exists():
            v = _load_json(r["ragas_run"])
            out["ragas_seconds"] = v.get("ragas_seconds")
            out["ragas_usd"] = v.get("ragas_usd")
            for cor in v.get("correlations", []):
                if cor["pair_name"].startswith("recall"):
                    out["ragas_context_recall"] = cor["ragas_mean"]
                    out["rho_recall"] = cor["spearman_rho"]
                    out["mae_recall"] = cor["mae"]
                elif cor["pair_name"].startswith("precision"):
                    out["ragas_context_precision"] = cor["ragas_mean"]
                    out["rho_precision"] = cor["spearman_rho"]
                    out["mae_precision"] = cor["mae"]
        rows.append(out)

    # Write HTML scoreboard
    args.output.write_text(_render_html(rows), encoding="utf-8")
    print(f"wrote {args.output}")

    # Append a Markdown table to MASTER_COMPARISON.md (rewriting §6).
    md_path = args.output.parent / "MASTER_COMPARISON.md"
    if md_path.exists():
        md = md_path.read_text(encoding="utf-8")
        # replace §6 table
        before, sep, after = md.partition("## 6 · Runs log")
        if sep:
            after = after.split("\n---\n", 1)
            tail = "\n---\n" + after[1] if len(after) > 1 else ""
            new_section = ["## 6 · Runs log — every experiment in one table\n",
                "> Latest at the bottom. Each row points to a self-contained `Runs/<id>.md`.\n",
                "| Run id | Retriever | top_k | n | text_recall_l3 | fact_recall (L1) | strict_l13 | fact_prec_rw | RAGAS ctx_recall | RAGAS ctx_prec | ρ recall | ρ prec | MAE recall | MAE prec | RAG_GT s | RAGAS s | RAGAS USD | File |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
            for r in rows:
                file_link = f"[[Runs/{r.get('file','')}]]"
                new_section.append(_table_row(
                    r["name"], r["retriever"], r["top_k"], r["n"],
                    r.get("text_recall_l3", 0), r.get("fact_recall", 0),
                    r.get("strict_recall_l13", 0), r.get("fact_precision_rw", 0),
                    r.get("ragas_context_recall"), r.get("ragas_context_precision"),
                    r.get("rho_recall"), r.get("rho_precision"),
                    r.get("mae_recall"), r.get("mae_precision"),
                    r.get("rag_gt_seconds", 0), r.get("ragas_seconds"),
                    r.get("ragas_usd"),
                    file_link,
                ))
            new_section.append("\n**How to add a new run:** copy `Runs/_TEMPLATE.md`, fill the front-matter and the headline numbers, then re-run `rag-gt-build-master-report` to refresh this table.\n")
            new_md = before + "\n".join(new_section) + tail
            md_path.write_text(new_md, encoding="utf-8")
            print(f"updated §6 of {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
