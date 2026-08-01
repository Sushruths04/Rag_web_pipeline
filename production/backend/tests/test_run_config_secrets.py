"""An API key supplied per-run must never be persisted.

The run store's rows are returned verbatim by GET /api/runs/{id}, so anything
written into `config` is readable by anyone with UI access. A key typed into
the Runs panel therefore has to reach the worker without going through the
store.
"""
from __future__ import annotations

from app.orchestrator.manager import SECRET_CONFIG_KEYS, split_secrets


def test_api_key_is_split_out_of_the_persistable_config():
    safe, secrets = split_secrets(
        {"llm_mode": "live", "max_cost_usd": 5.0, "api_key": "sk-super-secret"}
    )
    assert "api_key" not in safe
    assert secrets["api_key"] == "sk-super-secret"
    assert safe["llm_mode"] == "live"
    assert safe["max_cost_usd"] == 5.0


def test_supplying_a_key_is_recorded_without_the_key():
    safe, _ = split_secrets({"api_key": "sk-x"})
    assert safe["credentials_source"] == "run_config"
    assert "sk-x" not in repr(safe)


def test_no_secrets_means_no_marker_and_no_change():
    safe, secrets = split_secrets({"llm_mode": "import"})
    assert secrets == {}
    assert safe == {"llm_mode": "import"}
    assert "credentials_source" not in safe


def test_blank_key_is_not_treated_as_a_secret():
    """An empty field means 'use the server's key', not 'set the key to ""'."""
    safe, secrets = split_secrets({"api_key": "", "llm_mode": "live"})
    assert secrets == {}
    assert "credentials_source" not in safe


def test_base_url_is_also_treated_as_sensitive():
    assert "api_base_url" in SECRET_CONFIG_KEYS
    safe, secrets = split_secrets({"api_base_url": "https://private.internal/v1"})
    assert "api_base_url" not in safe
    assert secrets["api_base_url"] == "https://private.internal/v1"


def test_manager_does_not_persist_the_key(tmp_path):
    """End-to-end through the real store: the key must not come back out."""
    from app.db.store import Store
    from app.orchestrator.events import EventBus
    from app.orchestrator.manager import RunManager

    store = Store(tmp_path / "t.db")
    mgr = RunManager(store, EventBus(), tmp_path, mode="thread")
    run_id = mgr.prepare_run(
        {"llm_mode": "live", "api_key": "sk-must-not-persist"}, pipeline="dummy"
    )

    row = store.get_run(run_id)
    assert "sk-must-not-persist" not in repr(row)
    assert "api_key" not in row["config"]
    assert row["config"]["credentials_source"] == "run_config"
