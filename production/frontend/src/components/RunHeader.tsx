import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { cancelRun } from '../lib/api'
import type { RunState } from '../lib/runReducer'
import type { WsStatus } from '../lib/ws'
import StatusChip from './StatusChip'

interface Props {
  run: RunState
  createdAt: string
  wsStatus: WsStatus
}

export default function RunHeader({ run, createdAt, wsStatus }: Props) {
  const [, tick] = useState(0)
  const running = run.status === 'running' || run.status === 'queued'

  useEffect(() => {
    if (!running) return
    const t = setInterval(() => tick((n) => n + 1), 1000)
    return () => clearInterval(t)
  }, [running])

  const elapsedS = Math.max(0, (Date.now() - new Date(createdAt).getTime()) / 1000)
  const mm = Math.floor(elapsedS / 60)
  const ss = Math.floor(elapsedS % 60)
  const pairs = Object.values(run.stages).reduce<number | null>(
    (acc, s) => (s.metrics.pairs_kept != null ? s.metrics.pairs_kept : acc),
    null,
  )

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        padding: '12px 16px',
        borderBottom: '1px solid var(--line-soft)',
        flexWrap: 'wrap',
      }}
    >
      <Link to="/studio" className="dim" style={{ fontSize: 13 }}>← Runs</Link>
      <span className="mono" style={{ fontSize: 13 }}>{run.runId}</span>
      <StatusChip status={run.status} />
      <span
        title={wsStatus === 'live' ? 'live event stream' : 'reconnecting'}
        aria-label={`stream ${wsStatus}`}
        style={{
          width: 8,
          height: 8,
          borderRadius: '50%',
          background: wsStatus === 'live' ? 'var(--ok)' : 'var(--warn)',
          display: 'inline-block',
        }}
      />
      <span className="mono dim" style={{ fontSize: 12 }}>
        {running ? `elapsed ${mm}:${String(ss).padStart(2, '0')}` : ''}
      </span>
      <span style={{ marginLeft: 'auto', display: 'flex', gap: 14, alignItems: 'center' }}>
        {pairs != null && (
          <span className="metric-chip">pairs <b>{pairs}</b></span>
        )}
        <span className="metric-chip">LLM cost <b>${run.totalCostUsd.toFixed(4)}</b></span>
        {running && (
          <button className="danger" onClick={() => cancelRun(run.runId)}>
            Cancel run
          </button>
        )}
      </span>
    </div>
  )
}
