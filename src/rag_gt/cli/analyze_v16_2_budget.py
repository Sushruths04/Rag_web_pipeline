"""Compare V16.1 vs V16.2 build_summary.json cost/yield metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _output_path_from_summary_path(summary_path: str | Path) -> Path:
    path = Path(summary_path)
    suffix = ".build_summary.json"
    name = path.name
    if name.endswith(suffix):
        return path.with_name(name[: -len(suffix)])
    return path


def _doc_ids(summary: dict) -> list[str]:
    cost = summary.get("cost_tracker", {}) or {}
    cascade = summary.get("cascade_stats", {}) or {}
    ids = (set(cost) | set(cascade)) - {"aggregate"}
    return sorted(ids)


def _cost(summary: dict, doc_id: str) -> dict:
    return ((summary.get("cost_tracker") or {}).get(doc_id) or {})


def _yield(summary: dict, doc_id: str) -> dict:
    return (((summary.get("cascade_stats") or {}).get(doc_id) or {}).get("yield") or {})


def _accepted_strict_from_output(output_path: Path, doc_id: str | None = None) -> int:
    if not output_path.exists():
        return 0
    if doc_id is None:
        total = 0
        with output_path.open("r", encoding="utf-8") as f:
            for raw in f:
                if raw.strip():
                    total += 1
        return total
    count = 0
    with output_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if doc_id in set(row.get("doc_ids") or []):
                count += 1
    return count


def _accepted_strict(summary: dict, doc_id: str, summary_path: str | Path | None = None) -> int:
    y = _yield(summary, doc_id)
    for key in ("strict_total", "accepted_strict", "accepted"):
        if y.get(key) is not None:
            return int(y.get(key) or 0)
    if summary_path is not None:
        output_path = _output_path_from_summary_path(summary_path)
        if doc_id == "aggregate":
            return _accepted_strict_from_output(output_path, None)
        return _accepted_strict_from_output(output_path, doc_id)
    # Legacy fallback: if only aggregate run stats exist, leave per-doc unknown.
    if doc_id == "aggregate":
        total = 0
        for did in _doc_ids(summary):
            total += _accepted_strict(summary, did)
        return total
    return 0


def _live_per_strict(summary: dict, doc_id: str, summary_path: str | Path | None = None) -> float:
    c = _cost(summary, doc_id)
    accepted = _accepted_strict(summary, doc_id, summary_path=summary_path)
    if accepted <= 0:
        return 0.0
    return float(c.get("live_api_calls", 0) or 0) / accepted


def _cache_hit_share(summary: dict, doc_id: str) -> float:
    c = _cost(summary, doc_id)
    total = float(c.get("total_logical_calls", 0) or 0)
    if total <= 0:
        return 0.0
    return float(c.get("cache_hit_calls", 0) or 0) / total


def _typed_mh_share(summary: dict, doc_id: str) -> float:
    y = _yield(summary, doc_id)
    return float(y.get("typed_mh_share", 0.0) or 0.0)


def _aggregate_summary(summary: dict, summary_path: str | Path | None = None) -> dict:
    docs = _doc_ids(summary)
    live = sum(int(_cost(summary, did).get("live_api_calls", 0) or 0) for did in docs)
    cache = sum(int(_cost(summary, did).get("cache_hit_calls", 0) or 0) for did in docs)
    total = sum(int(_cost(summary, did).get("total_logical_calls", 0) or 0) for did in docs)
    accepted = sum(_accepted_strict(summary, did, summary_path=summary_path) for did in docs)
    typed = sum(
        int((_yield(summary, did).get("typed_mh") or {}).get("accepted", 0) or 0)
        for did in docs
    )
    return {
        "live_api_calls": live,
        "cache_hit_calls": cache,
        "total_logical_calls": total,
        "accepted_strict": accepted,
        "live_per_strict": (live / accepted) if accepted else 0.0,
        "typed_mh_share": (typed / accepted) if accepted else 0.0,
        "cache_hit_share": (cache / total) if total else 0.0,
    }


def _row(
    doc_id: str,
    v16_1: dict,
    v16_2: dict,
    v16_1_path: str | Path | None = None,
    v16_2_path: str | Path | None = None,
) -> dict[str, Any]:
    v1_live_per = _live_per_strict(v16_1, doc_id, summary_path=v16_1_path)
    v2_live_per = _live_per_strict(v16_2, doc_id, summary_path=v16_2_path)
    reduction = ((v1_live_per - v2_live_per) / v1_live_per) if v1_live_per else 0.0
    return {
        "doc": doc_id,
        "v16_1_live_per_strict": v1_live_per,
        "v16_2_live_per_strict": v2_live_per,
        "reduction": reduction,
        "v16_2_typed_mh_share": _typed_mh_share(v16_2, doc_id),
        "v16_2_cache_hit_share": _cache_hit_share(v16_2, doc_id),
    }


def build_markdown(
    v16_1: dict,
    v16_2: dict,
    v16_1_path: str | Path | None = None,
    v16_2_path: str | Path | None = None,
) -> str:
    docs = sorted(set(_doc_ids(v16_1)) | set(_doc_ids(v16_2)))
    rows = [_row(doc_id, v16_1, v16_2, v16_1_path=v16_1_path, v16_2_path=v16_2_path) for doc_id in docs]

    agg1 = _aggregate_summary(v16_1, summary_path=v16_1_path)
    agg2 = _aggregate_summary(v16_2, summary_path=v16_2_path)
    agg_reduction = (
        (agg1["live_per_strict"] - agg2["live_per_strict"]) / agg1["live_per_strict"]
        if agg1["live_per_strict"]
        else 0.0
    )
    rows.append({
        "doc": "aggregate",
        "v16_1_live_per_strict": agg1["live_per_strict"],
        "v16_2_live_per_strict": agg2["live_per_strict"],
        "reduction": agg_reduction,
        "v16_2_typed_mh_share": agg2["typed_mh_share"],
        "v16_2_cache_hit_share": agg2["cache_hit_share"],
    })

    lines = [
        "| doc | V16.1 live_api_calls / accepted_strict | V16.2 live_api_calls / accepted_strict | live-call reduction | V16.2 typed_mh_share | V16.2 cache_hit_share |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['doc']}` | {r['v16_1_live_per_strict']:.2f} | "
            f"{r['v16_2_live_per_strict']:.2f} | {r['reduction']:.1%} | "
            f"{r['v16_2_typed_mh_share']:.1%} | {r['v16_2_cache_hit_share']:.1%} |"
        )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag-gt-analyze-v16-2-budget",
        description="Compare V16.1 and V16.2 build_summary.json cost/yield metrics.",
    )
    p.add_argument("--v16_1-summary", required=True)
    p.add_argument("--v16_2-summary", required=True)
    p.add_argument("--out", default=None, help="Optional Markdown output path.")
    return p


def main() -> int:
    args = _parser().parse_args()
    md = build_markdown(
        _load(args.v16_1_summary),
        _load(args.v16_2_summary),
        v16_1_path=args.v16_1_summary,
        v16_2_path=args.v16_2_summary,
    )
    print(md)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
