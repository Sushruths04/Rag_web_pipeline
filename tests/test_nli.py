"""Tests for NLI cache semantics and batch behavior."""

import importlib
import sqlite3

from tests.helpers import cleanup_workspace_tmp, make_workspace_tmp


def test_cache_key_separates_pairs_with_shared_prefix(monkeypatch):
    """The cache key now hashes the FULL premise+hypothesis (the previous
    pre-hash truncation caused two long, distinct pairs sharing a 1200/400-char
    prefix to collide and overwrite each other's scores)."""
    tmpdir = make_workspace_tmp("nli")
    try:
        monkeypatch.setenv("NLI_CACHE_PATH", str(tmpdir / "nli_cache.db"))
        import rag_gt.validation.nli_check as mod

        importlib.reload(mod)
        key1 = mod._cache_key("a" * 1500, "b" * 500)
        key2 = mod._cache_key("a" * 1200 + "tail", "b" * 400 + "tail")
        assert key1 != key2  # they only shared a prefix; different content

        monkeypatch.setattr(mod, "NLI_MODEL_NAME", "other-model")
        key3 = mod._cache_key("a" * 1500, "b" * 500)
        assert key3 != key1  # model is part of the key
    finally:
        cleanup_workspace_tmp(tmpdir)


def test_cache_key_stable_for_identical_inputs(monkeypatch):
    tmpdir = make_workspace_tmp("nli")
    try:
        monkeypatch.setenv("NLI_CACHE_PATH", str(tmpdir / "nli_cache.db"))
        import rag_gt.validation.nli_check as mod

        importlib.reload(mod)
        a = mod._cache_key("premise text", "hypothesis text")
        b = mod._cache_key("premise text", "hypothesis text")
        assert a == b
    finally:
        cleanup_workspace_tmp(tmpdir)


def test_batch_answer_entailment_skips_insufficient(monkeypatch):
    tmpdir = make_workspace_tmp("nli")
    try:
        monkeypatch.setenv("NLI_CACHE_PATH", str(tmpdir / "nli_cache.db"))
        import rag_gt.validation.nli_check as mod
        from rag_gt.core.types import Fact

        importlib.reload(mod)
        calls = []

        def fake_nli_batch(pairs):
            calls.append(pairs)
            return [0.9 for _ in pairs]

        monkeypatch.setattr(mod, "nli_batch", fake_nli_batch)
        facts = [Fact(fact_id="F1", text="A long enough supporting fact.", role="rule")]
        out = mod.batch_answer_entailment(
            [
                ("Insufficient information to answer.", facts),
                ("This answer is grounded in the facts.", facts),
            ]
        )

        assert out == [True, True]
        assert len(calls) == 1
        assert len(calls[0]) == 1
    finally:
        cleanup_workspace_tmp(tmpdir)


def test_nli_entailment_uses_cache(monkeypatch):
    tmpdir = make_workspace_tmp("nli")
    try:
        db_path = tmpdir / "nli_cache.db"
        monkeypatch.setenv("NLI_CACHE_PATH", str(db_path))
        import rag_gt.core.models as core_models
        import rag_gt.validation.nli_check as mod

        importlib.reload(mod)

        fake_model = type(
            "FakeModel",
            (),
            {"predict": lambda self, pairs, apply_softmax=True: [[0.1, 0.8, 0.1] for _ in pairs]},
        )()
        monkeypatch.setattr(mod.MM, "get_nli", lambda: fake_model)
        monkeypatch.setattr(mod.MM, "load_nli", lambda: None)
        # Resolve labels by name (mirrors what core.models would do).
        core_models.NLI_LABEL_INDEX.clear()
        core_models.NLI_LABEL_INDEX.update(
            {"contradiction": 0, "entailment": 1, "neutral": 2}
        )

        score1 = mod.nli_entailment("premise", "hypothesis")
        score2 = mod.nli_entailment("premise", "hypothesis")
        assert abs(score1 - 0.8) < 1e-6
        assert score1 == score2

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT COUNT(*) FROM nli_cache").fetchone()[0]
        conn.close()
        assert rows == 1
    finally:
        cleanup_workspace_tmp(tmpdir)
