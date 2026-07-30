"""API response schemas. The frontend TypeScript types mirror these 1:1."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class StageInfo(BaseModel):
    name: str
    label: str
    deps: list[str]
    track: str


class StageState(BaseModel):
    name: str
    status: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_s: Optional[float] = None
    error: Optional[str] = None
    traceback: Optional[str] = None


class RunSummary(BaseModel):
    run_id: str
    pipeline: str
    status: str
    created_at: str
    error: Optional[str] = None


class RunDetail(RunSummary):
    config: dict[str, Any]
    stages: list[StageState]
    metrics: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
