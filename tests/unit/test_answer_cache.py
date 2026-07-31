import pytest
from tests.helpers import cleanup_workspace_tmp, make_workspace_tmp


@pytest.fixture
def tmp_cache(monkeypatch):
    tmpdir = make_workspace_tmp("answer_cache")
    try:
        db = tmpdir / "answer_cache.db"
        monkeypatch.setenv("ANSWER_CACHE_PATH", str(db))
        import importlib

        import rag_gt.storage.cache as mod

        importlib.reload(mod)
        yield mod
    finally:
        cleanup_workspace_tmp(tmpdir)


def test_key_stable_for_same_inputs(tmp_cache):
    k1 = tmp_cache.make_key("gpt-4", "What is X?", ["b", "a", "c"])
    k2 = tmp_cache.make_key("gpt-4", "What is X?", ["a", "b", "c"])
    assert k1 == k2


def test_put_and_get(tmp_cache):
    key = tmp_cache.make_key("m", "q", ["f1"])
    assert tmp_cache.get(key) is None
    tmp_cache.put(key, "hello world answer", "m")
    assert tmp_cache.get(key) == "hello world answer"


def test_put_overwrites_second_write(tmp_cache):
    """The cache uses INSERT OR REPLACE so a deliberate re-put (e.g. after
    fixing a prompt) updates the cached answer. The previous semantics
    (INSERT OR IGNORE) silently dropped the new value."""
    key = tmp_cache.make_key("m", "q", ["f1"])
    tmp_cache.put(key, "first", "m")
    tmp_cache.put(key, "second", "m")
    assert tmp_cache.get(key) == "second"


def test_make_key_dedupes_repeated_fact_ids(tmp_cache):
    a = tmp_cache.make_key("m", "q", ["f1", "f1", "f2"])
    b = tmp_cache.make_key("m", "q", ["f1", "f2"])
    assert a == b


def test_make_key_no_collision_for_pipe_in_question(tmp_cache):
    """The previous string-join key collided when the question contained "||".
    The JSON-encoded key must not."""
    a = tmp_cache.make_key("m", "A||B", ["x"])
    b = tmp_cache.make_key("m||A", "B", ["x"])
    assert a != b


def test_key_differs_across_models(tmp_cache):
    k1 = tmp_cache.make_key("modelA", "q", ["f1"])
    k2 = tmp_cache.make_key("modelB", "q", ["f1"])
    assert k1 != k2
