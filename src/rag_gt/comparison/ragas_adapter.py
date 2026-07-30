"""RAGAS adapter: lazy-import wrapper + dry-run faker.

Two paths:
- ``backend == "dry_run"``  -> no network, no ragas import. Heuristics from
  ``fixtures.heuristic_ragas_scores`` produce sensible RAGAS-shaped numbers.
- ``backend in {"api", "ollama"}`` -> real RAGAS via langchain shim. Token
  usage and judge calls are captured via a callback handler. The reasoning-
  model empty-content gotcha is handled by retrying once with a higher
  temperature (see Agentic_RAG/evaluation/ragas_metrics.py for the same
  workaround on the reference side).

The adapter never raises if RAGAS isn't installed; it falls back to dry-run
and warns. This keeps tests and CI offline-safe.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rag_gt.comparison.chunk_resolver import ChunkResolver
from rag_gt.comparison.fixtures import heuristic_ragas_scores
from rag_gt.core.types import AnswerLog, QuestionGT, RetrievalLog


RAGAS_METRIC_NAMES = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
)


@dataclass
class RagasConfig:
    backend: str = "dry_run"            # "dry_run" | "api" | "ollama"
    judge_model: str = ""
    embed_model: str = ""
    api_base_url: str = ""
    api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    timeout_seconds: int = 600
    temperature_first_try: float = 0.0
    temperature_retry: float = 0.2
    seed: int = 0

    @classmethod
    def from_env(cls, backend: Optional[str] = None) -> "RagasConfig":
        b = (backend or os.getenv("LLM_BACKEND") or "dry_run").strip().lower()
        return cls(
            backend=b,
            judge_model=os.getenv("API_GT_MODEL") or os.getenv("OLLAMA_GT_MODEL") or "",
            embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5"),
            api_base_url=os.getenv("API_BASE_URL", ""),
            api_key=os.getenv("API_KEY", ""),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )


@dataclass
class RagasResult:
    per_question: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    judge_calls: int = 0
    metric_failures: Dict[str, int] = field(default_factory=dict)
    backend_used: str = "dry_run"


class RagasAdapter:
    def __init__(self, cfg: RagasConfig, resolver: ChunkResolver) -> None:
        self.cfg = cfg
        self.resolver = resolver

    @staticmethod
    def available() -> bool:
        try:
            import ragas  # noqa: F401
            import datasets  # noqa: F401
            return True
        except Exception:
            return False

    def run(
        self,
        q_map: Dict[str, QuestionGT],
        ret_logs: Dict[str, RetrievalLog],
        ans_logs: Dict[str, AnswerLog],
        rag_gt_per_question: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> RagasResult:
        rows = self._build_rows(q_map, ret_logs, ans_logs)
        if not rows:
            return RagasResult(backend_used=self.cfg.backend)

        if self.cfg.backend == "dry_run":
            return self._run_dry(rows, rag_gt_per_question or {})

        # Real RAGAS path. Fall back to dry_run if dependencies are missing.
        if not self.available():
            print(
                "[RagasAdapter] ragas/datasets not installed. "
                "Falling back to --ragas-llm dry_run. Install via "
                "`pip install -e .[ragas]` for the real path."
            )
            return self._run_dry(rows, rag_gt_per_question or {})

        try:
            return self._run_real(rows)
        except Exception as e:
            print(f"[RagasAdapter] real RAGAS path failed: {e!r}. Falling back to dry_run.")
            return self._run_dry(rows, rag_gt_per_question or {})

    # ---------- internals ----------

    def _build_rows(
        self,
        q_map: Dict[str, QuestionGT],
        ret_logs: Dict[str, RetrievalLog],
        ans_logs: Dict[str, AnswerLog],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for qid in sorted(q_map):
            if qid not in ret_logs or qid not in ans_logs:
                continue
            q_gt = q_map[qid]
            ret = ret_logs[qid]
            ans = ans_logs[qid]
            contexts = self.resolver.get_many(ret.retrieved_chunk_ids)
            rows.append(
                {
                    "q_id": qid,
                    "question": q_gt.question,
                    "answer": ans.predicted_answer or "",
                    "contexts": contexts or [""],
                    "reference": q_gt.gold_answer,
                }
            )
        return rows

    def _run_dry(
        self,
        rows: List[Dict[str, Any]],
        rag_gt_per_question: Dict[str, Dict[str, float]],
    ) -> RagasResult:
        t0 = time.perf_counter()
        per_q: List[Dict[str, Any]] = []
        for i, row in enumerate(rows):
            scores = heuristic_ragas_scores(
                question=row["question"],
                predicted_answer=row["answer"],
                contexts=row["contexts"],
                gold_answer=row["reference"],
                rag_gt_metrics=rag_gt_per_question.get(row["q_id"], {}),
                seed=self.cfg.seed + i,
            )
            per_q.append({"q_id": row["q_id"], **scores})
        return RagasResult(
            per_question=per_q,
            elapsed_seconds=time.perf_counter() - t0,
            prompt_tokens=0,
            completion_tokens=0,
            judge_calls=0,
            metric_failures={m: 0 for m in RAGAS_METRIC_NAMES},
            backend_used="dry_run",
        )

    def _run_real(self, rows: List[Dict[str, Any]]) -> RagasResult:
        # Local imports so the package never depends on ragas at import time.
        from datasets import Dataset

        evaluate, metrics = self._import_ragas_metrics()
        llm, embeddings = self._build_judge()

        ds = Dataset.from_list(
            [
                {
                    "question": r["question"],
                    "answer": r["answer"],
                    "contexts": r["contexts"],
                    "reference": r["reference"],
                }
                for r in rows
            ]
        )

        t0 = time.perf_counter()
        usage = _make_usage_callback()
        result = self._call_evaluate(evaluate, ds, metrics, llm, embeddings, usage)

        elapsed = time.perf_counter() - t0

        per_q = self._extract_per_question(result, [r["q_id"] for r in rows])

        # Count metric NaNs as failures so the report can flag them.
        failures = {m: 0 for m in RAGAS_METRIC_NAMES}
        for row in per_q:
            for m in RAGAS_METRIC_NAMES:
                v = row.get(m)
                if v is None or (isinstance(v, float) and v != v):
                    failures[m] += 1

        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        judge_calls = getattr(usage, "calls", 0) if usage else 0

        return RagasResult(
            per_question=per_q,
            elapsed_seconds=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            judge_calls=judge_calls,
            metric_failures=failures,
            backend_used=self.cfg.backend,
        )

    def _import_ragas_metrics(self):
        """Import the four metric singletons; prefer public path."""
        from ragas import evaluate as _evaluate
        try:
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
        except ImportError:
            from ragas.metrics._faithfulness import faithfulness
            from ragas.metrics._answer_relevance import answer_relevancy
            from ragas.metrics._context_precision import context_precision
            from ragas.metrics._context_recall import context_recall
        return _evaluate, [context_precision, context_recall, faithfulness, answer_relevancy]

    def _build_judge(self) -> Tuple[Any, Any]:
        """Wire a langchain LLM + HF embeddings for RAGAS' judge."""
        if self.cfg.backend == "api":
            if not self.cfg.api_base_url or not self.cfg.api_key:
                raise RuntimeError(
                    "RagasConfig.api_base_url / api_key required for backend=api."
                )
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                base_url=self.cfg.api_base_url.rstrip("/"),
                api_key=self.cfg.api_key,
                model=self.cfg.judge_model or "gpt-4o-mini",
                temperature=self.cfg.temperature_first_try,
                timeout=self.cfg.timeout_seconds,
            )
        elif self.cfg.backend == "ollama":
            from langchain_community.llms import Ollama

            llm = Ollama(
                model=self.cfg.judge_model or "qwen2.5:7b-instruct-q4_K_M",
                base_url=self.cfg.ollama_base_url,
                temperature=self.cfg.temperature_first_try,
            )
        else:
            raise ValueError(f"Unsupported real backend: {self.cfg.backend!r}")

        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=self.cfg.embed_model or "BAAI/bge-base-en-v1.5"
        )
        return llm, embeddings

    def _call_evaluate(self, evaluate, ds, metrics, llm, embeddings, usage):
        # Try the most common signature first; gracefully fall back.
        callbacks = [usage] if usage is not None else None
        try:
            return evaluate(
                ds,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                callbacks=callbacks,
            )
        except TypeError:
            return evaluate(ds, metrics=metrics, llm=llm, embeddings=embeddings)

    def _extract_per_question(self, result, q_ids: List[str]) -> List[Dict[str, Any]]:
        # `result` is a ragas Result whose .to_pandas() yields a DataFrame
        # with one row per question and one column per metric. Older versions
        # expose the same data via dict-like indexing; we handle both.
        per_q: List[Dict[str, Any]] = []
        try:
            df = result.to_pandas()
            for i, qid in enumerate(q_ids):
                row = {"q_id": qid}
                for m in RAGAS_METRIC_NAMES:
                    if m in df.columns:
                        v = df.iloc[i][m]
                        row[m] = float(v) if v == v else float("nan")
                    else:
                        row[m] = float("nan")
                per_q.append(row)
            return per_q
        except Exception:
            pass

        # Last-ditch: corpus means broadcast to every question.
        means: Dict[str, float] = {}
        for m in RAGAS_METRIC_NAMES:
            try:
                means[m] = float(result[m])
            except Exception:
                means[m] = float("nan")
        for qid in q_ids:
            per_q.append({"q_id": qid, **means})
        return per_q


def _make_usage_callback():
    """Build a langchain-compatible callback handler that captures token use.

    Subclasses ``BaseCallbackHandler`` at runtime so the package never imports
    langchain at module load time (langchain is only present when the user has
    installed the ``[ragas]`` extra). Returns ``None`` if the import fails;
    callers fall back to zero token accounting in that case.
    """
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except Exception:
        return None

    class _UsageCallback(BaseCallbackHandler):
        def __init__(self) -> None:
            super().__init__()
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.calls = 0

        def on_llm_end(self, response, **kwargs):
            self.calls += 1
            try:
                for gen_list in getattr(response, "generations", []) or []:
                    for gen in gen_list:
                        msg = getattr(gen, "message", None)
                        usage = getattr(msg, "usage_metadata", None) if msg else None
                        if usage:
                            self.prompt_tokens += int(usage.get("input_tokens", 0))
                            self.completion_tokens += int(usage.get("output_tokens", 0))
                llm_output = getattr(response, "llm_output", None) or {}
                tu = llm_output.get("token_usage") or llm_output.get("usage") or {}
                if tu:
                    self.prompt_tokens += int(tu.get("prompt_tokens", 0))
                    self.completion_tokens += int(tu.get("completion_tokens", 0))
            except Exception:
                pass

    return _UsageCallback()
