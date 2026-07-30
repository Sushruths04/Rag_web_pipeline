"""Drive both evaluators side-by-side and compute correlation statistics.

Reuses ``cli/evaluate._evaluate_corpus_batched`` for the RAG_GT side — never
reimplements the metric logic. The RAGAS side runs through ``RagasAdapter``.
Per-question scores are aligned by ``q_id``; correlations are computed only on
the metric pairs we explicitly map (FSR<->context_recall, FSP<->context_
precision, faithfulness<->faithfulness). Other metrics are surfaced as
"RAG_GT only" / "RAGAS only" in the report.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rag_gt.comparison.chunk_resolver import ChunkResolver, _normalized_key
from rag_gt.comparison.cost_tracker import CostReport, CostTracker
from rag_gt.comparison.ragas_adapter import (
    RAGAS_METRIC_NAMES,
    RagasAdapter,
    RagasConfig,
    RagasResult,
)
from rag_gt.core.types import AnswerLog, QuestionGT, RetrievalLog


# (rag_gt_metric, ragas_metric, human_label)
METRIC_PAIRS: List[Tuple[str, str, str]] = [
    ("fact_span_recall", "context_recall", "FSR vs context_recall"),
    ("fact_span_precision", "context_precision", "FSP vs context_precision"),
    ("faithfulness", "faithfulness", "NLI faithfulness vs RAGAS faithfulness"),
]
RAG_GT_ONLY_METRICS = (
    "kpr",
    "fact_precision",
    "contradiction_rate",
    "abstention_correct",
    "hallucination_flag",
)
RAGAS_ONLY_METRICS = ("answer_relevancy",)


@dataclass
class PerQuestionRow:
    q_id: str
    rag_gt: Dict[str, float] = field(default_factory=dict)
    ragas: Dict[str, float] = field(default_factory=dict)


@dataclass
class CorrelationCell:
    pair_label: str
    rag_gt_metric: str
    ragas_metric: str
    n: int
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    kendall_tau: float
    kendall_p: float

    def to_dict(self) -> Dict[str, float | str | int]:
        return {
            "pair_label": self.pair_label,
            "rag_gt_metric": self.rag_gt_metric,
            "ragas_metric": self.ragas_metric,
            "n": self.n,
            "pearson_r": self.pearson_r,
            "pearson_p": self.pearson_p,
            "spearman_rho": self.spearman_rho,
            "spearman_p": self.spearman_p,
            "kendall_tau": self.kendall_tau,
            "kendall_p": self.kendall_p,
        }


@dataclass
class ComparisonReport:
    rows: List[PerQuestionRow] = field(default_factory=list)
    correlations: List[CorrelationCell] = field(default_factory=list)
    rag_gt_corpus: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ragas_corpus: Dict[str, Dict[str, float]] = field(default_factory=dict)
    rank_agreement: float = float("nan")
    cost: CostReport = field(default_factory=CostReport)
    config: Dict[str, object] = field(default_factory=dict)
    ragas_metric_failures: Dict[str, int] = field(default_factory=dict)
    ragas_backend_used: str = "dry_run"

    def to_dict(self) -> Dict[str, object]:
        return {
            "config": self.config,
            "rag_gt_corpus": self.rag_gt_corpus,
            "ragas_corpus": self.ragas_corpus,
            "rank_agreement": self.rank_agreement,
            "ragas_backend_used": self.ragas_backend_used,
            "ragas_metric_failures": self.ragas_metric_failures,
            "correlations": [c.to_dict() for c in self.correlations],
            "cost": self.cost.to_dict(),
            "rows": [
                {"q_id": r.q_id, "rag_gt": r.rag_gt, "ragas": r.ragas}
                for r in self.rows
            ],
        }


class Comparator:
    def __init__(
        self,
        gt_path: Path | str,
        retrieval_path: Path | str,
        answers_path: Path | str,
        resolver: ChunkResolver,
        ragas_cfg: RagasConfig,
        cost_tracker: CostTracker,
        limit: Optional[int] = None,
        skip_rag_gt: bool = False,
        skip_ragas: bool = False,
    ) -> None:
        self.gt_path = Path(gt_path)
        self.retrieval_path = Path(retrieval_path)
        self.answers_path = Path(answers_path)
        self.resolver = resolver
        self.ragas_cfg = ragas_cfg
        self.cost = cost_tracker
        self.limit = limit
        self.skip_rag_gt = skip_rag_gt
        self.skip_ragas = skip_ragas

    def run(self) -> ComparisonReport:
        q_map, ret_logs, ans_logs = self._load_inputs()
        matched = sorted(q for q in q_map if q in ret_logs and q in ans_logs)
        if self.limit is not None:
            matched = matched[: self.limit]

        if not matched:
            raise RuntimeError(
                "No matching q_ids across GT, retrieval, and answer logs."
            )

        q_map_m = {qid: q_map[qid] for qid in matched}
        ret_m = {qid: ret_logs[qid] for qid in matched}
        ans_m = {qid: ans_logs[qid] for qid in matched}

        self.cost.set_n_questions(len(matched))

        rag_gt_per_q = self._run_rag_gt(q_map_m, ret_m, ans_m)
        ragas_result = self._run_ragas(q_map_m, ret_m, ans_m, rag_gt_per_q)

        rows = self._merge(matched, rag_gt_per_q, ragas_result)
        correlations = self._compute_correlations(rows)
        rg_corpus = self._corpus_stats(rows, side="rag_gt")
        rs_corpus = self._corpus_stats(rows, side="ragas")
        rank_agreement = self._rank_agreement(rows)

        return ComparisonReport(
            rows=rows,
            correlations=correlations,
            rag_gt_corpus=rg_corpus,
            ragas_corpus=rs_corpus,
            rank_agreement=rank_agreement,
            cost=self.cost.report,
            config={
                "gt": str(self.gt_path),
                "retrieval": str(self.retrieval_path),
                "answers": str(self.answers_path),
                "limit": self.limit,
                "ragas_backend": self.ragas_cfg.backend,
                "ragas_judge_model": self.ragas_cfg.judge_model,
                "ragas_embed_model": self.ragas_cfg.embed_model,
                "skip_rag_gt": self.skip_rag_gt,
                "skip_ragas": self.skip_ragas,
            },
            ragas_metric_failures=ragas_result.metric_failures,
            ragas_backend_used=ragas_result.backend_used,
        )

    # ---------- I/O ----------

    def _load_inputs(
        self,
    ) -> Tuple[Dict[str, QuestionGT], Dict[str, RetrievalLog], Dict[str, AnswerLog]]:
        # Reuse storage.gt_io.load_gt by deriving corpus_name from filename.
        from rag_gt.storage.gt_io import load_gt

        corpus_name = self.gt_path.stem
        gt_dir = self.gt_path.parent or None
        questions = load_gt(corpus_name, in_dir=gt_dir)
        q_map = {q.q_id: q for q in questions}

        ret_logs: Dict[str, RetrievalLog] = {}
        with open(self.retrieval_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                d = json.loads(line)
                ret_logs[d["q_id"]] = RetrievalLog(
                    q_id=d["q_id"], retrieved_chunk_ids=d["retrieved_chunk_ids"]
                )

        ans_logs: Dict[str, AnswerLog] = {}
        with open(self.answers_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                d = json.loads(line)
                ans_logs[d["q_id"]] = AnswerLog(
                    q_id=d["q_id"],
                    predicted_answer=d.get("predicted_answer", ""),
                    abstained=d.get("abstained", False),
                )
        return q_map, ret_logs, ans_logs

    # ---------- RAG_GT side ----------

    def _run_rag_gt(
        self,
        q_map: Dict[str, QuestionGT],
        ret_logs: Dict[str, RetrievalLog],
        ans_logs: Dict[str, AnswerLog],
    ) -> Dict[str, Dict[str, float]]:
        if self.skip_rag_gt:
            return {qid: {} for qid in q_map}
        from rag_gt.cli.evaluate import _evaluate_corpus_batched, METRIC_KEYS

        # Normalise chunk_id zero-padding so width drift between GT-generation
        # time (4-digit) and current chunker (6-digit) does not silently zero
        # out FSR/FSP. Affects only the in-memory copies passed to the
        # evaluator; on-disk GT and logs are unchanged.
        q_map_n = self._normalize_chunk_ids_in_gt(q_map)
        ret_n = self._normalize_chunk_ids_in_logs(ret_logs)

        # Phase 4 (plan v5): pass the resolver into the evaluator so fact-span
        # metrics use source-span projection into the active chunk profile.
        # This is what makes V11 retrieval logs (which use a different chunk
        # profile than the GT was generated against) score correctly instead
        # of silently zeroing out.
        with self.cost.stage("rag_gt"):
            rows = _evaluate_corpus_batched(q_map_n, ret_n, ans_logs, resolver=self.resolver)
        # _evaluate_corpus_batched returns rows in the same order as the
        # matched_ids it computed; reproduce the alignment by re-deriving the
        # order it uses (intersection in dict iteration order of q_map).
        matched_ids = [qid for qid in q_map if qid in ret_logs and qid in ans_logs]
        out: Dict[str, Dict[str, float]] = {}
        for qid, row in zip(matched_ids, rows):
            out[qid] = {k: float(row[k]) for k in METRIC_KEYS}
        return out

    @staticmethod
    def _normalize_chunk_ids_in_gt(
        q_map: Dict[str, QuestionGT],
    ) -> Dict[str, QuestionGT]:
        from copy import deepcopy

        out: Dict[str, QuestionGT] = {}
        for qid, q in q_map.items():
            qc = deepcopy(q)
            for f in qc.required_facts:
                for s in f.supporting_spans:
                    s.chunk_id = _normalized_key(s.chunk_id)
            out[qid] = qc
        return out

    @staticmethod
    def _normalize_chunk_ids_in_logs(
        ret_logs: Dict[str, RetrievalLog],
    ) -> Dict[str, RetrievalLog]:
        out: Dict[str, RetrievalLog] = {}
        for qid, rl in ret_logs.items():
            out[qid] = RetrievalLog(
                q_id=rl.q_id,
                retrieved_chunk_ids=[_normalized_key(c) for c in rl.retrieved_chunk_ids],
            )
        return out

    # ---------- RAGAS side ----------

    def _run_ragas(
        self,
        q_map: Dict[str, QuestionGT],
        ret_logs: Dict[str, RetrievalLog],
        ans_logs: Dict[str, AnswerLog],
        rag_gt_per_q: Dict[str, Dict[str, float]],
    ) -> RagasResult:
        if self.skip_ragas:
            return RagasResult(backend_used="skipped")
        adapter = RagasAdapter(self.ragas_cfg, self.resolver)
        if self.ragas_cfg.judge_model:
            self.cost.set_judge_model(self.ragas_cfg.judge_model)

        with self.cost.stage("ragas"):
            result = adapter.run(q_map, ret_logs, ans_logs, rag_gt_per_q)
        self.cost.add_ragas_tokens(
            result.prompt_tokens, result.completion_tokens, result.judge_calls
        )
        return result

    # ---------- merge + stats ----------

    def _merge(
        self,
        matched: List[str],
        rag_gt_per_q: Dict[str, Dict[str, float]],
        ragas_result: RagasResult,
    ) -> List[PerQuestionRow]:
        ragas_by_id = {row["q_id"]: row for row in ragas_result.per_question}
        out: List[PerQuestionRow] = []
        for qid in matched:
            ragas_row = {
                m: float(ragas_by_id.get(qid, {}).get(m, float("nan")))
                for m in RAGAS_METRIC_NAMES
            }
            out.append(
                PerQuestionRow(
                    q_id=qid,
                    rag_gt=rag_gt_per_q.get(qid, {}),
                    ragas=ragas_row,
                )
            )
        return out

    def _compute_correlations(
        self, rows: List[PerQuestionRow]
    ) -> List[CorrelationCell]:
        out: List[CorrelationCell] = []
        for rg_metric, rs_metric, label in METRIC_PAIRS:
            xs: List[float] = []
            ys: List[float] = []
            for r in rows:
                x = r.rag_gt.get(rg_metric, float("nan"))
                y = r.ragas.get(rs_metric, float("nan"))
                if _is_finite(x) and _is_finite(y):
                    xs.append(x)
                    ys.append(y)
            cell = _correlation_cell(label, rg_metric, rs_metric, xs, ys)
            out.append(cell)
        return out

    def _corpus_stats(
        self, rows: List[PerQuestionRow], side: str
    ) -> Dict[str, Dict[str, float]]:
        if side == "rag_gt":
            from rag_gt.cli.evaluate import METRIC_KEYS as keys
        else:
            keys = list(RAGAS_METRIC_NAMES)
        stats: Dict[str, Dict[str, float]] = {}
        for k in keys:
            vals = [
                getattr(r, side).get(k, float("nan")) for r in rows
            ]
            cleaned = [v for v in vals if _is_finite(v)]
            if not cleaned:
                stats[k] = {
                    "mean": float("nan"),
                    "median": float("nan"),
                    "std": float("nan"),
                    "n": 0,
                }
                continue
            stats[k] = {
                "mean": _mean(cleaned),
                "median": _median(cleaned),
                "std": _stdev(cleaned),
                "n": len(cleaned),
            }
        return stats

    def _rank_agreement(self, rows: List[PerQuestionRow]) -> float:
        """Composite rank-agreement: z-score and average each side's three
        shared metrics, then Spearman over the per-question composites."""
        rag_vec: List[float] = []
        rs_vec: List[float] = []
        for r in rows:
            rag = _composite(r.rag_gt, [m[0] for m in METRIC_PAIRS])
            rs = _composite(r.ragas, [m[1] for m in METRIC_PAIRS])
            if _is_finite(rag) and _is_finite(rs):
                rag_vec.append(rag)
                rs_vec.append(rs)
        if len(rag_vec) < 3:
            return float("nan")
        rho, _ = _spearman(rag_vec, rs_vec)
        return rho


# ---------- math helpers (no scipy if not available) ----------


def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and x == x and not math.isinf(x)


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def _median(xs: List[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def _stdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def _composite(d: Dict[str, float], keys: List[str]) -> float:
    vals = [d.get(k, float("nan")) for k in keys]
    cleaned = [v for v in vals if _is_finite(v)]
    if not cleaned:
        return float("nan")
    return sum(cleaned) / len(cleaned)


def _correlation_cell(
    label: str, rg_metric: str, rs_metric: str, xs: List[float], ys: List[float]
) -> CorrelationCell:
    n = len(xs)
    if n < 3:
        return CorrelationCell(
            pair_label=label,
            rag_gt_metric=rg_metric,
            ragas_metric=rs_metric,
            n=n,
            pearson_r=float("nan"),
            pearson_p=float("nan"),
            spearman_rho=float("nan"),
            spearman_p=float("nan"),
            kendall_tau=float("nan"),
            kendall_p=float("nan"),
        )
    p_r, p_p = _pearson(xs, ys)
    s_rho, s_p = _spearman(xs, ys)
    k_tau, k_p = _kendall(xs, ys)
    return CorrelationCell(
        pair_label=label,
        rag_gt_metric=rg_metric,
        ragas_metric=rs_metric,
        n=n,
        pearson_r=p_r,
        pearson_p=p_p,
        spearman_rho=s_rho,
        spearman_p=s_p,
        kendall_tau=k_tau,
        kendall_p=k_p,
    )


def _try_scipy():
    try:
        from scipy import stats as _stats
        return _stats
    except Exception:
        return None


def _pearson(xs, ys) -> Tuple[float, float]:
    s = _try_scipy()
    if s is not None:
        r, p = s.pearsonr(xs, ys)
        return float(r), float(p)
    n = len(xs)
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(dx2 * dy2) if dx2 and dy2 else 0.0
    r = num / denom if denom else float("nan")
    return float(r), float("nan")  # p-value omitted in fallback


def _spearman(xs, ys) -> Tuple[float, float]:
    s = _try_scipy()
    if s is not None:
        rho, p = s.spearmanr(xs, ys)
        return float(rho), float(p)
    rx = _ranks(xs)
    ry = _ranks(ys)
    return _pearson(rx, ry)


def _kendall(xs, ys) -> Tuple[float, float]:
    s = _try_scipy()
    if s is not None:
        tau, p = s.kendalltau(xs, ys)
        return float(tau), float(p)
    n = len(xs)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            sign = dx * dy
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    total = n * (n - 1) / 2
    tau = (concordant - discordant) / total if total else float("nan")
    return float(tau), float("nan")


def _ranks(values: List[float]) -> List[float]:
    """Average-rank tied values."""
    indexed = sorted(enumerate(values), key=lambda t: t[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks
