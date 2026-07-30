import type { PipelineEvent, RunDetail, RunStatus, StageStatus } from './types'

export interface StageRuntime {
  status: StageStatus
  progress?: { done: number; total: number; message: string }
  durationS?: number
  error?: string
  traceback?: string
  metrics: Record<string, number>
  logCount: number
}

export interface RunState {
  runId: string
  status: RunStatus
  lastSeq: number
  stages: Record<string, StageRuntime>
  totalCostUsd: number
  logs: PipelineEvent[]
}

const LOG_CAP = 1000

export function initRunState(detail: RunDetail): RunState {
  const stages: Record<string, StageRuntime> = {}
  for (const s of detail.stages) {
    stages[s.name] = {
      status: s.status,
      durationS: s.duration_s ?? undefined,
      error: s.error ?? undefined,
      traceback: s.traceback ?? undefined,
      metrics: {},
      logCount: 0,
    }
  }
  let totalCostUsd = 0
  for (const m of detail.metrics) {
    const st = stages[m.stage]
    if (st) st.metrics[m.name] = m.value
    if (m.name === 'llm_cost_usd') totalCostUsd += m.value
  }
  return {
    runId: detail.run_id,
    status: detail.status,
    lastSeq: 0,
    stages,
    totalCostUsd,
    logs: [],
  }
}

function blankStage(): StageRuntime {
  return { status: 'queued', metrics: {}, logCount: 0 }
}

export function runReducer(state: RunState, ev: PipelineEvent): RunState {
  if (ev.seq <= state.lastSeq) return state

  const next: RunState = { ...state, lastSeq: ev.seq }

  const patchStage = (name: string, patch: Partial<StageRuntime>) => {
    const prev = next.stages[name] ?? blankStage()
    next.stages = { ...next.stages, [name]: { ...prev, ...patch } }
  }

  const stage = ev.stage
  switch (ev.type) {
    case 'stage_started':
      if (stage) patchStage(stage, { status: 'running', error: undefined, traceback: undefined, progress: undefined })
      break
    case 'stage_progress':
      if (stage)
        patchStage(stage, {
          progress: {
            done: Number(ev.payload.done ?? 0),
            total: Number(ev.payload.total ?? 0),
            message: String(ev.payload.message ?? ''),
          },
        })
      break
    case 'stage_log': {
      if (stage) {
        const prev = next.stages[stage] ?? blankStage()
        patchStage(stage, { logCount: prev.logCount + 1 })
      }
      const logs = state.logs.length >= LOG_CAP ? state.logs.slice(state.logs.length - LOG_CAP + 1) : state.logs.slice()
      logs.push(ev)
      next.logs = logs
      break
    }
    case 'stage_metric': {
      if (stage) {
        const prev = next.stages[stage] ?? blankStage()
        const name = String(ev.payload.name)
        const value = Number(ev.payload.value)
        patchStage(stage, { metrics: { ...prev.metrics, [name]: value } })
        if (name === 'llm_cost_usd') next.totalCostUsd = state.totalCostUsd + value
      }
      break
    }
    case 'stage_completed':
      if (stage) patchStage(stage, { status: 'completed', durationS: Number(ev.payload.duration_s ?? 0) })
      break
    case 'stage_failed':
      if (stage)
        patchStage(stage, {
          status: 'failed',
          error: ev.payload.error != null ? String(ev.payload.error) : undefined,
          traceback: ev.payload.traceback != null ? String(ev.payload.traceback) : undefined,
          durationS: ev.payload.duration_s != null ? Number(ev.payload.duration_s) : undefined,
        })
      break
    case 'stage_skipped':
      if (stage) patchStage(stage, { status: 'skipped' })
      break
    case 'run_completed':
      next.status = 'completed'
      break
    case 'run_failed':
      next.status = 'failed'
      break
    case 'run_cancelled':
      next.status = 'cancelled'
      break
  }
  return next
}
