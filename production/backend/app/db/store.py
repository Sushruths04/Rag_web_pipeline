"""SQLite persistence layer. Single writer: only the API process writes.

Thread-safe via one connection + RLock (uvicorn threads + queue pump thread).
JSON columns (config, payload, input, output, extra) are serialized here and
deserialized on read, so callers only ever see dicts.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_SCHEMA = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _loads(v: Optional[str]) -> Any:
    return json.loads(v) if v else None


class Store:
    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- runs ---------------------------------------------------------------

    def create_run(self, run_id: str, config: dict, pipeline: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (run_id, pipeline, status, config, created_at) VALUES (?,?,?,?,?)",
                (run_id, pipeline, "queued", json.dumps(config), _now()),
            )
            self._conn.commit()

    def set_run_status(self, run_id: str, status: str, error: Optional[str] = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET status=?, error=? WHERE run_id=?", (status, error, run_id)
            )
            self._conn.commit()

    def get_run(self, run_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                return None
            stages = self._conn.execute(
                "SELECT * FROM stages WHERE run_id=? ORDER BY rowid", (run_id,)
            ).fetchall()
        run = dict(row)
        run["config"] = _loads(run["config"]) or {}
        run["stages"] = [dict(s) for s in stages]
        return run

    def list_runs(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["config"] = _loads(d["config"]) or {}
            out.append(d)
        return out

    # -- stages ---------------------------------------------------------------

    _STAGE_FIELDS = ("status", "started_at", "finished_at", "duration_s", "error", "traceback")

    def upsert_stage(self, run_id: str, name: str, **fields: Any) -> None:
        bad = set(fields) - set(self._STAGE_FIELDS)
        if bad:
            raise ValueError(f"unknown stage fields: {bad}")
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO stages (run_id, name) VALUES (?,?)", (run_id, name)
            )
            if fields:
                sets = ", ".join(f"{k}=?" for k in fields)
                self._conn.execute(
                    f"UPDATE stages SET {sets} WHERE run_id=? AND name=?",
                    (*fields.values(), run_id, name),
                )
            self._conn.commit()

    # -- events ---------------------------------------------------------------

    def append_event(self, event: dict) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (run_id, stage, type, ts, payload) VALUES (?,?,?,?,?)",
                (
                    event["run_id"],
                    event.get("stage"),
                    event["type"],
                    event.get("ts") or _now(),
                    json.dumps(event.get("payload") or {}),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def events_after(
        self,
        run_id: str,
        after_seq: int = 0,
        stage: Optional[str] = None,
        type: Optional[str] = None,
        limit: int = 2000,
    ) -> list[dict]:
        q = "SELECT * FROM events WHERE run_id=? AND seq>?"
        args: list[Any] = [run_id, after_seq]
        if stage is not None:
            q += " AND stage=?"
            args.append(stage)
        if type is not None:
            q += " AND type=?"
            args.append(type)
        q += " ORDER BY seq LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = _loads(d["payload"]) or {}
            out.append(d)
        return out

    # -- metrics / artifacts ----------------------------------------------------

    def add_metric(self, run_id: str, stage: str, name: str, value: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO metrics (run_id, stage, name, value, ts) VALUES (?,?,?,?,?)",
                (run_id, stage, name, float(value), _now()),
            )
            self._conn.commit()

    def list_metrics(self, run_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM metrics WHERE run_id=? ORDER BY rowid", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def add_artifact(
        self, run_id: str, stage: str, name: str, path: str, kind: str, size_bytes: int
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO artifacts (run_id, stage, name, path, kind, size_bytes, ts) VALUES (?,?,?,?,?,?,?)",
                (run_id, stage, name, path, kind, size_bytes, _now()),
            )
            self._conn.commit()

    def list_artifacts(self, run_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM artifacts WHERE run_id=? ORDER BY rowid", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- spans ---------------------------------------------------------------

    _SPAN_JSON = ("input", "output", "extra")

    def add_span(self, span: dict) -> None:
        s = dict(span)
        for k in self._SPAN_JSON:
            if s.get(k) is not None:
                s[k] = json.dumps(s[k])
        cols = (
            "span_id,run_id,stage,parent_id,type,name,status,started_at,"
            "duration_ms,input,output,tokens_in,tokens_out,cost_usd,extra"
        )
        with self._lock:
            self._conn.execute(
                f"INSERT INTO spans ({cols}) VALUES ({','.join('?' * 15)})",
                tuple(s.get(c) for c in cols.split(",")),
            )
            self._conn.commit()

    def list_spans(
        self,
        run_id: str,
        stage: Optional[str] = None,
        type: Optional[str] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[dict]:
        q = "SELECT * FROM spans WHERE run_id=?"
        args: list[Any] = [run_id]
        if stage is not None:
            q += " AND stage=?"
            args.append(stage)
        if type is not None:
            q += " AND type=?"
            args.append(type)
        q += " ORDER BY started_at LIMIT ? OFFSET ?"
        args += [limit, offset]
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in self._SPAN_JSON:
                d[k] = _loads(d[k])
            out.append(d)
        return out
