import { Handle, Position } from '@xyflow/react'
import type { StageRuntime } from '../lib/runReducer'

export interface StageNodeData {
  label: string
  track: 'gt' | 'rag' | 'eval'
  runtime?: StageRuntime
  [key: string]: unknown
}

const TRACK_VAR: Record<string, string> = {
  gt: 'var(--track-gt)',
  rag: 'var(--track-rag)',
  eval: 'var(--track-eval)',
}

function fmtDuration(s?: number) {
  if (s == null) return ''
  return s >= 60 ? `${Math.floor(s / 60)}m ${(s % 60).toFixed(0)}s` : `${s.toFixed(1)}s`
}

export default function StageNode({ data }: { data: StageNodeData }) {
  const rt = data.runtime
  const status = rt?.status ?? 'queued'
  const pct =
    status === 'completed' || status === 'failed'
      ? 100
      : rt?.progress && rt.progress.total > 0
        ? Math.round((100 * rt.progress.done) / rt.progress.total)
        : 0

  return (
    <div className={`stage-node ${status}`} style={{ '--track': TRACK_VAR[data.track] } as React.CSSProperties}>
      <Handle type="target" position={Position.Left} style={{ opacity: 0 }} />
      <div className="track-tag">{data.track}</div>
      <div className="label">{data.label}</div>
      <div className="meta">
        {status === 'running' && (
          <>
            <span className="spinner" aria-label="running" />
            <span>{rt?.progress ? `${pct}% · ${rt.progress.message}` : 'starting…'}</span>
          </>
        )}
        {status === 'completed' && (
          <>
            <span className="badge-ok">✓</span>
            <span>{fmtDuration(rt?.durationS)}</span>
          </>
        )}
        {status === 'failed' && (
          <>
            <span className="badge-err">✕</span>
            <span style={{ color: 'var(--err)' }}>failed — open for traceback</span>
          </>
        )}
        {status === 'skipped' && <span>skipped</span>}
        {status === 'queued' && <span className="faint">queued</span>}
      </div>
      <div className="ribbon" aria-hidden>
        <div style={{ width: `${pct}%` }} />
      </div>
      <Handle type="source" position={Position.Right} style={{ opacity: 0 }} />
    </div>
  )
}
