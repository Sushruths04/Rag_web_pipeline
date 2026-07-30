"""Deterministically re-score a finished run's final-eval rows with the
current matching metrics (precision_rw / f1_rw). Reads only run artifacts;
no retrieval, no LLM, never mutates the run.

Usage (from production/backend/):
    python scripts/rescore_run.py ../data/runs/<run_id> --out ../../data/eval_results/RESCORE_<run_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.matching import aggregate, score_pair  # noqa: E402


def rescore(run_dir: Path) -> dict:
    art = run_dir / "artifacts"
    chunk_texts: dict[str, str] = {}
    for f in sorted(art.glob("chunks_*.json")):
        for c in json.loads(f.read_text(encoding="utf-8")):
            chunk_texts[c["chunk_id"]] = c.get("text", "")
    pairs = {}
    for line in (art / "gt.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)
            pairs[p["qa_id"]] = p
    old = json.loads((art / "eval_results.json").read_text(encoding="utf-8"))
    rows = []
    missing_chunks = 0
    for r in old["rows"]:
        texts = []
        for cid in r["retrieved"]:
            if cid in chunk_texts:
                texts.append(chunk_texts[cid])
            else:
                missing_chunks += 1
        rows.append(score_pair(pairs[r["qa_id"]], texts))
    return {
        "old": old["aggregate"],
        "new": aggregate(rows),
        "config": old["winner"]["config"],
        "n_rows": len(rows),
        "missing_chunks": missing_chunks,
    }


def _fmt_md(run_id: str, out: dict) -> str:
    o, n = out["old"], out["new"]
    lines = [
        f"# Deterministic re-score — run {run_id} ({out['config']})",
        "",
        "Recorded retrievals re-scored with rank-weighted precision. No retrieval",
        "or LLM was run; identical inputs, fairer metric.",
        "",
        "| metric | old | new |",
        "|---|---:|---:|",
        f"| recall | {o['recall']:.4f} | {n['recall']:.4f} |",
        f"| precision (raw @k) | {o['precision']:.4f} | {n['precision']:.4f} |",
        f"| precision_rw | — | {n['precision_rw']:.4f} |",
        f"| f1 (raw) | {o['f1']:.4f} | {n['f1']:.4f} |",
        f"| f1_rw | — | {n['f1_rw']:.4f} |",
        "",
        "## Per hop (new)",
        "",
        "| hop | n | recall | precision_rw | raw precision |",
        "|---|---:|---:|---:|---:|",
    ]
    for hop, d in n["by_hop"].items():
        lines.append(
            f"| {hop} | {d['n']} | {d['recall']:.4f} | {d['precision_rw']:.4f} | {d['precision']:.4f} |"
        )
    lines.append("")
    lines.append(f"Rows: {out['n_rows']}; retrieved chunk ids not found in artifacts: {out['missing_chunks']}.")
    lines.append("")
    lines.append("Note: only the final-eval config's retrievals are recorded in the run;")
    lines.append("sweep cells are re-ranked live on the next Studio run (local, free).")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="output basename (writes .json and .md)")
    args = ap.parse_args()
    out = rescore(args.run_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    args.out.with_suffix(".md").write_text(_fmt_md(args.run_dir.name, out), encoding="utf-8")
    print(json.dumps({"old_precision": out["old"]["precision"],
                      "new_precision_rw": out["new"]["precision_rw"],
                      "new_f1_rw": out["new"]["f1_rw"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
