"""Track wall time + token cost for the comparison harness.

RAG_GT side cost is always $0 (no LLM API calls — local NLI + lexical match).
RAGAS side cost is computed from token counts captured by the adapter using a
small embedded price table (USD per 1k tokens). The price table can be
overridden via JSON at the CLI.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional


# USD per 1k tokens. "unknown" forces price=0 and surfaces a warning in the
# Markdown report when the user's judge model is not in the table.
_DEFAULT_PRICE_TABLE: Dict[str, Dict[str, float]] = {
    "gpt-4o":         {"prompt": 0.00250, "completion": 0.01000},
    "gpt-4o-mini":    {"prompt": 0.00015, "completion": 0.00060},
    "gpt-4-turbo":    {"prompt": 0.01000, "completion": 0.03000},
    "gpt-3.5-turbo":  {"prompt": 0.00050, "completion": 0.00150},
    "openai/gpt-oss-20b":  {"prompt": 0.00020, "completion": 0.00060},
    "openai/gpt-oss-120b": {"prompt": 0.00040, "completion": 0.00150},
    "unknown":        {"prompt": 0.00000, "completion": 0.00000},
}


def _load_price_table(path: Optional[Path | str]) -> Dict[str, Dict[str, float]]:
    if path is None:
        return dict(_DEFAULT_PRICE_TABLE)
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        custom = json.load(f)
    merged = dict(_DEFAULT_PRICE_TABLE)
    merged.update(custom)
    return merged


@dataclass
class StageTiming:
    name: str
    seconds: float = 0.0
    notes: str = ""


@dataclass
class CostReport:
    rag_gt_seconds: float = 0.0
    rag_gt_usd: float = 0.0
    rag_gt_api_calls: int = 0
    ragas_seconds: float = 0.0
    ragas_prompt_tokens: int = 0
    ragas_completion_tokens: int = 0
    ragas_judge_calls: int = 0
    ragas_usd: float = 0.0
    judge_model: str = ""
    price_table_source: str = "default"
    price_unknown: bool = False
    n_questions: int = 0

    @property
    def usd_per_question_rag_gt(self) -> float:
        return self.rag_gt_usd / self.n_questions if self.n_questions else 0.0

    @property
    def usd_per_question_ragas(self) -> float:
        return self.ragas_usd / self.n_questions if self.n_questions else 0.0

    @property
    def speedup(self) -> float:
        if self.rag_gt_seconds <= 0:
            return float("nan")
        return self.ragas_seconds / self.rag_gt_seconds

    def to_dict(self) -> Dict[str, float]:
        return {
            "rag_gt_seconds": self.rag_gt_seconds,
            "rag_gt_usd": self.rag_gt_usd,
            "rag_gt_api_calls": self.rag_gt_api_calls,
            "ragas_seconds": self.ragas_seconds,
            "ragas_prompt_tokens": self.ragas_prompt_tokens,
            "ragas_completion_tokens": self.ragas_completion_tokens,
            "ragas_judge_calls": self.ragas_judge_calls,
            "ragas_usd": self.ragas_usd,
            "judge_model": self.judge_model,
            "price_table_source": self.price_table_source,
            "price_unknown": self.price_unknown,
            "n_questions": self.n_questions,
            "usd_per_question_ragas": self.usd_per_question_ragas,
            "speedup_rag_gt_over_ragas": self.speedup,
        }


class CostTracker:
    def __init__(
        self,
        price_table_path: Optional[Path | str] = None,
        judge_model: str = "",
    ) -> None:
        self.price_table = _load_price_table(price_table_path)
        self.price_table_source = (
            str(price_table_path) if price_table_path else "default"
        )
        self._stages: Dict[str, StageTiming] = {}
        self.report = CostReport(
            judge_model=judge_model,
            price_table_source=self.price_table_source,
        )

    @contextmanager
    def stage(self, name: str) -> Iterator[StageTiming]:
        t0 = time.perf_counter()
        st = StageTiming(name=name)
        self._stages[name] = st
        try:
            yield st
        finally:
            st.seconds = time.perf_counter() - t0
            if name == "rag_gt":
                self.report.rag_gt_seconds = st.seconds
            elif name == "ragas":
                self.report.ragas_seconds = st.seconds

    def add_ragas_tokens(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        judge_calls: int,
    ) -> None:
        self.report.ragas_prompt_tokens += int(prompt_tokens)
        self.report.ragas_completion_tokens += int(completion_tokens)
        self.report.ragas_judge_calls += int(judge_calls)
        self._recompute_usd()

    def set_judge_model(self, model: str) -> None:
        self.report.judge_model = model
        self._recompute_usd()

    def _recompute_usd(self) -> None:
        model = self.report.judge_model or "unknown"
        prices = self.price_table.get(model)
        if prices is None:
            prices = self.price_table["unknown"]
            self.report.price_unknown = True
        self.report.ragas_usd = (
            self.report.ragas_prompt_tokens / 1000.0 * prices["prompt"]
            + self.report.ragas_completion_tokens / 1000.0 * prices["completion"]
        )

    def set_n_questions(self, n: int) -> None:
        self.report.n_questions = int(n)

    def to_json(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.report.to_dict(), f, ensure_ascii=False, indent=2)
        return p
