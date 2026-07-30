import { describe, expect, it } from 'vitest'
import type { PipelineEvent, RunDetail } from './types'
import { initRunState, runReducer, type RunState } from './runReducer'

const detail: RunDetail = {
  run_id: 'r1',
  pipeline: 'dummy',
  status: 'running',
  created_at: '2026-07-05T10:00:00Z',
  error: null,
  config: {},
  stages: [
    { name: 'ingest', status: 'completed', duration_s: 1.2 },
    { name: 'chunk', status: 'running' },
    { name: 'extract', status: 'queued' },
  ],
  metrics: [
    { run_id: 'r1', stage: 'verify', name: 'llm_cost_usd', value: 0.5, ts: 't' },
    { run_id: 'r1', stage: 'extract', name: 'facts_extracted', value: 100, ts: 't' },
  ],
  artifacts: [],
}

function ev(seq: number, type: PipelineEvent['type'], stage: string | null, payload: Record<string, unknown> = {}): PipelineEvent {
  return { seq, run_id: 'r1', stage, type, ts: 't', payload }
}

function init(): RunState {
  return initRunState(detail)
}

describe('initRunState', () => {
  it('maps snapshot stages, metrics, and cost', () => {
    const s = init()
    expect(s.runId).toBe('r1')
    expect(s.status).toBe('running')
    expect(s.stages.ingest.status).toBe('completed')
    expect(s.stages.ingest.durationS).toBe(1.2)
    expect(s.stages.extract.metrics.facts_extracted).toBe(100)
    expect(s.totalCostUsd).toBeCloseTo(0.5)
    expect(s.lastSeq).toBe(0)
  })
})

describe('runReducer', () => {
  it('applies the stage lifecycle', () => {
    let s = init()
    s = runReducer(s, ev(1, 'stage_started', 'extract', { label: 'Fact Extraction' }))
    expect(s.stages.extract.status).toBe('running')
    s = runReducer(s, ev(2, 'stage_progress', 'extract', { done: 3, total: 10, message: 'batch 3/10' }))
    expect(s.stages.extract.progress).toEqual({ done: 3, total: 10, message: 'batch 3/10' })
    s = runReducer(s, ev(3, 'stage_metric', 'extract', { name: 'facts_extracted', value: 512 }))
    expect(s.stages.extract.metrics.facts_extracted).toBe(512)
    s = runReducer(s, ev(4, 'stage_completed', 'extract', { duration_s: 8.5 }))
    expect(s.stages.extract.status).toBe('completed')
    expect(s.stages.extract.durationS).toBe(8.5)
    expect(s.lastSeq).toBe(4)
  })

  it('accumulates llm_cost_usd into totalCostUsd', () => {
    let s = init()
    s = runReducer(s, ev(1, 'stage_metric', 'verify', { name: 'llm_cost_usd', value: 1.25 }))
    expect(s.totalCostUsd).toBeCloseTo(1.75)
  })

  it('captures failure with error + traceback and skips', () => {
    let s = init()
    s = runReducer(s, ev(1, 'stage_failed', 'chunk', { error: 'ValueError: bad', traceback: 'tb...' }))
    expect(s.stages.chunk.status).toBe('failed')
    expect(s.stages.chunk.error).toBe('ValueError: bad')
    expect(s.stages.chunk.traceback).toBe('tb...')
    s = runReducer(s, ev(2, 'stage_skipped', 'extract', { reason: 'upstream_failed' }))
    expect(s.stages.extract.status).toBe('skipped')
  })

  it('sets terminal run status', () => {
    let s = init()
    s = runReducer(s, ev(1, 'run_failed', null))
    expect(s.status).toBe('failed')
    s = runReducer(s, ev(2, 'run_completed', null))
    expect(s.status).toBe('completed')
  })

  it('ignores duplicate or stale events', () => {
    let s = init()
    s = runReducer(s, ev(5, 'stage_started', 'extract'))
    const before = s
    s = runReducer(s, ev(5, 'stage_failed', 'extract', { error: 'x' }))
    expect(s).toBe(before)
    s = runReducer(s, ev(3, 'stage_failed', 'extract', { error: 'x' }))
    expect(s).toBe(before)
  })

  it('collects stage logs and caps the ring buffer at 1000', () => {
    let s = init()
    for (let i = 1; i <= 1005; i++) {
      s = runReducer(s, ev(i, 'stage_log', 'chunk', { level: 'info', line: `l${i}` }))
    }
    expect(s.logs.length).toBe(1000)
    expect(s.logs[0].payload.line).toBe('l6')
    expect(s.stages.chunk.logCount).toBe(1005)
  })

  it('creates unknown stages defensively', () => {
    let s = init()
    s = runReducer(s, ev(1, 'stage_started', 'mystery'))
    expect(s.stages.mystery.status).toBe('running')
  })
})
