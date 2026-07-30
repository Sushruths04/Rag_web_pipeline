"""Write a compact Markdown review sheet for a GT JSONL file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Iterable


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _norm_question(text: str) -> str:
    return " ".join((text or "").split()).lower()


def _word_count(text: str) -> int:
    return len((text or "").split())


def _avg(values: Iterable[int]) -> float:
    values = list(values)
    return mean(values) if values else 0.0


def _md_escape(text: str, limit: int = 180) -> str:
    text = " ".join((text or "").split())
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text.replace("|", "\\|")


def _counter_table(title: str, counts: Counter) -> list[str]:
    lines = [f"\n## {title}", "", "| Value | Count |", "|---|---:|"]
    for key, value in sorted(counts.items(), key=lambda kv: str(kv[0])):
        lines.append(f"| `{key}` | {value} |")
    if not counts:
        lines.append("| n/a | 0 |")
    return lines


def _write_markdown(
    rows: list[dict],
    output: Path,
    label: str,
    gt_path: Path,
    drop_stats: dict,
    max_examples: int,
) -> None:
    doc_counts = Counter(doc for r in rows for doc in r.get("doc_ids", []))
    depth_counts = Counter(r.get("difficulty_reasoning_depth", "missing") for r in rows)
    distance_counts = Counter(
        r.get("difficulty_semantic_distance", "missing") for r in rows
    )
    role_counts = Counter(
        f.get("role", "missing")
        for r in rows
        for f in r.get("required_facts", [])
    )
    q_norms = [_norm_question(r.get("question", "")) for r in rows]
    duplicate_questions = len(q_norms) - len(set(q_norms))
    facts_per_q = [len(r.get("required_fact_ids", [])) for r in rows]
    answer_words = [_word_count(r.get("gold_answer", "")) for r in rows]

    lines = [
        f"# GT Review: {label}",
        "",
        f"- Source JSONL: `{gt_path}`",
        f"- Questions: **{len(rows)}**",
        f"- Documents covered: **{len(doc_counts)}**",
        f"- Duplicate normalized questions: **{duplicate_questions}**",
        f"- Avg required facts/question: **{_avg(facts_per_q):.2f}**",
        f"- Avg gold-answer words: **{_avg(answer_words):.1f}**",
    ]

    lines.extend(_counter_table("Documents", doc_counts))
    lines.extend(_counter_table("Reasoning Depth", depth_counts))
    lines.extend(_counter_table("Semantic Distance", distance_counts))
    lines.extend(_counter_table("Required Fact Roles", role_counts))

    if drop_stats:
        lines.extend(
            [
                "\n## Drop Stats",
                "",
                "| Metric | Count |",
                "|---|---:|",
            ]
        )
        for key, value in drop_stats.items():
            if key == "per_depth":
                continue
            lines.append(f"| `{key}` | {value} |")
        if drop_stats.get("per_depth"):
            lines.extend(["", "| Depth | Kept | Minimality Drops |", "|---:|---:|---:|"])
            for depth, stats in sorted(
                drop_stats["per_depth"].items(), key=lambda kv: int(kv[0])
            ):
                lines.append(
                    f"| {depth} | {stats.get('kept', 0)} | "
                    f"{stats.get('minimality', 0)} |"
                )

    lines.extend(
        [
            "\n## Review Examples",
            "",
            "| # | q_id | Depth | Question | Gold Answer | Required Facts |",
            "|---:|---|---:|---|---|---|",
        ]
    )
    for idx, row in enumerate(rows[:max_examples], start=1):
        facts = "<br>".join(
            _md_escape(f.get("text", ""), limit=110)
            for f in row.get("required_facts", [])[:4]
        )
        lines.append(
            f"| {idx} | `{row.get('q_id', '')}` | "
            f"{row.get('difficulty_reasoning_depth', '')} | "
            f"{_md_escape(row.get('question', ''))} | "
            f"{_md_escape(row.get('gold_answer', ''))} | {facts} |"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag-gt-summarize",
        description="Summarize a GT JSONL file into Markdown for review.",
    )
    p.add_argument("--gt", required=True, help="Ground-truth JSONL path.")
    p.add_argument("--output", required=True, help="Markdown output path.")
    p.add_argument("--label", default=None, help="Report label.")
    p.add_argument("--drop_stats", default=None, help="Optional drop-stats JSON.")
    p.add_argument("--max_examples", type=int, default=25)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    gt_path = Path(args.gt)
    output = Path(args.output)
    drop_stats_path = Path(args.drop_stats) if args.drop_stats else None
    rows = _load_jsonl(gt_path)
    _write_markdown(
        rows=rows,
        output=output,
        label=args.label or gt_path.stem,
        gt_path=gt_path,
        drop_stats=_load_json(drop_stats_path),
        max_examples=args.max_examples,
    )
    print(f"[GT] Wrote review markdown -> {output}")


if __name__ == "__main__":
    main()
