"""Persistent SQLite answer cache for LOO minimality calls.

Key = sha256(json([model, question, sorted(set(fact_ids))])). Survives across
runs, so the same leave-one-out regeneration costs nothing the second time.

Behaviour notes:
- Connections are thread-local (with WAL mode) because minimality checks run
  leave-one-out calls concurrently. Sharing one sqlite3.Connection across
  worker threads was observed to raise intermittent InterfaceError failures.
- `put` uses `INSERT OR REPLACE` so a deliberate overwrite (after fixing a
  prompt) updates the cached answer; relying on first-write-wins is fragile.
- `make_key` JSON-encodes the inputs, so model names, questions, or fact_ids
  containing `||` or `,` no longer collide.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

from rag_gt.core.config import load_config, repo_root

_cfg = load_config()
_DEFAULT_PATH = _cfg.get("performance", {}).get(
    "answer_cache_path", "data/cache/answer_cache.db"
)
_CACHE_PATH = Path(
    os.getenv("ANSWER_CACHE_PATH") or str(repo_root() / _DEFAULT_PATH)
)

_conn_lock = threading.Lock()
_local = threading.local()
_key_locks: dict[str, threading.Lock] = {}
_key_locks_guard = threading.Lock()
_KEY_LOCKS_MAX = 8192


def _get_key_lock(key: str) -> threading.Lock:
    with _key_locks_guard:
        lock = _key_locks.get(key)
        if lock is None:
            # Bound the lock dict to avoid unbounded memory growth on long runs.
            if len(_key_locks) >= _KEY_LOCKS_MAX:
                _key_locks.clear()
            lock = threading.Lock()
            _key_locks[key] = lock
        return lock


def _connect() -> sqlite3.Connection:
    """Return the current thread's SQLite connection."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        return conn
    with _conn_lock:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(_CACHE_PATH), timeout=30, check_same_thread=False
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS answer_cache (
                key TEXT PRIMARY KEY,
                answer TEXT NOT NULL,
                model TEXT,
                ts REAL
            )"""
        )
        conn.commit()
        _local.conn = conn
        return conn


def make_key(model: str, question: str, fact_ids: Iterable[str]) -> str:
    """Stable cache key. Order-independent in fact_ids (we sort + dedupe)."""
    payload = json.dumps(
        {
            "model": model,
            "question": question,
            "fact_ids": sorted(set(fact_ids)),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get(key: str) -> Optional[str]:
    conn = _connect()
    with _conn_lock:
        row = conn.execute(
            "SELECT answer FROM answer_cache WHERE key=?", (key,)
        ).fetchone()
    return row[0] if row else None


def put(key: str, answer: str, model: str) -> None:
    conn = _connect()
    with _conn_lock:
        conn.execute(
            "INSERT OR REPLACE INTO answer_cache (key, answer, model, ts) "
            "VALUES (?, ?, ?, ?)",
            (key, answer, model, time.time()),
        )
        conn.commit()


def key_lock(key: str) -> threading.Lock:
    return _get_key_lock(key)


def close() -> None:
    """Close this thread's connection. Useful for tests."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _local.conn = None
