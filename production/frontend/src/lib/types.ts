/** Mirrors backend/app/models.py and the event wire format exactly. */

export type RunStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
export type StageStatus = 'queued' | 'running' | 'completed' | 'failed' | 'skipped' | 'cancelled'
export type SpanType = 'llm' | 'retrieval' | 'processing' | 'embedding'
export type EventType =
  | 'stage_started' | 'stage_progress' | 'stage_log' | 'stage_metric'
  | 'stage_completed' | 'stage_failed' | 'stage_skipped'
  | 'run_completed' | 'run_failed' | 'run_cancelled'

export interface StageInfo {
  name: string
  label: string
  deps: string[]
  track: 'gt' | 'rag' | 'eval'
}

export interface StageState {
  name: string
  status: StageStatus
  started_at?: string | null
  finished_at?: string | null
  duration_s?: number | null
  error?: string | null
  traceback?: string | null
}

export interface RunSummary {
  run_id: string
  pipeline: string
  status: RunStatus
  created_at: string
  error?: string | null
}

export interface MetricRow {
  run_id: string
  stage: string
  name: string
  value: number
  ts: string
}

export interface ArtifactRow {
  run_id: string
  stage: string
  name: string
  path: string
  kind: string
  size_bytes: number
  ts: string
}

export interface RunDetail extends RunSummary {
  config: Record<string, unknown>
  stages: StageState[]
  metrics: MetricRow[]
  artifacts: ArtifactRow[]
}

export interface PipelineEvent {
  seq: number
  run_id: string
  stage: string | null
  type: EventType
  ts: string
  payload: Record<string, unknown>
}

export interface Span {
  span_id: string
  run_id: string
  stage: string
  parent_id: string | null
  type: SpanType
  name: string
  status: 'ok' | 'error'
  started_at: string
  duration_ms: number | null
  input: unknown
  output: unknown
  tokens_in: number | null
  tokens_out: number | null
  cost_usd: number | null
  extra: Record<string, unknown> | null
}
