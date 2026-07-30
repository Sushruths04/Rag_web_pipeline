import { useState } from 'react'
import { artifactUrl, retryStage } from '../lib/api'
import type { ArtifactRow, PipelineEvent, RunStatus } from '../lib/types'
import type { StageRuntime } from '../lib/runReducer'
import LogTail from './LogTail'
import StatusChip from './StatusChip'

interface Props {
  runId: string
  stage: string
  label: string
  runtime: StageRuntime
  runStatus: RunStatus
  logs: PipelineEvent[]
  artifacts: ArtifactRow[]
  onClose: () => void
  onViewSpans: () => void
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <div className="faint mono" style={{ fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>
        {title}
      </div>
      {children}
    </section>
  )
}

export default function NodeInspector({
  runId, stage, label, runtime, runStatus, logs, artifacts, onClose, onViewSpans,
}: Props) {
  const [showTb, setShowTb] = useState(false)
  const [retrying, setRetrying] = useState(false)
  const [retryError, setRetryError] = useState<string | null>(null)
  const stageLogs = logs.filter((l) => l.stage === stage)
  const runActive = runStatus === 'running' || runStatus === 'queued'

  const onRetry = async () => {
    setRetrying(true)
    setRetryError(null)
    try {
      await retryStage(runId, stage)
    } catch (e) {
      setRetryError(String(e))
    } finally {
      setRetrying(false)
    }
  }

  return (
    <aside className="drawer" aria-label={`${label} details`}>
      <header>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 0 }}>
          <b style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{label}</b>
          <StatusChip status={runtime.status} />
        </div>
        <button onClick={onClose} aria-label="Close">✕</button>
      </header>
      <div className="body">
        <div className="mono dim" style={{ fontSize: 12, display: 'flex', gap: 14, flexWrap: 'wrap' }}>
          {runtime.durationS != null && <span>duration {runtime.durationS.toFixed(1)}s</span>}
          {runtime.progress && runtime.status === 'running' && (
            <span>
              {runtime.progress.done}/{runtime.progress.total} {runtime.progress.message}
            </span>
          )}
          <span>{runtime.logCount} log lines</span>
        </div>

        {runtime.status === 'failed' && (
          <Section title="Failure">
            <div className="mono" style={{ color: 'var(--err)', fontSize: 12, marginBottom: 8 }}>{runtime.error}</div>
            {runtime.traceback && (
              <button onClick={() => setShowTb(!showTb)} style={{ marginBottom: 8 }}>
                {showTb ? 'Hide traceback' : 'Show traceback'}
              </button>
            )}
            {showTb && <pre className="json">{runtime.traceback}</pre>}
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <button className="primary" onClick={onRetry} disabled={retrying || runActive} title={runActive ? 'Wait for the run to stop' : undefined}>
                {retrying ? 'Retrying…' : 'Retry stage'}
              </button>
              {retryError && <span className="mono" style={{ color: 'var(--err)', fontSize: 11 }}>{retryError}</span>}
            </div>
          </Section>
        )}

        {Object.keys(runtime.metrics).length > 0 && (
          <Section title="Metrics">
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {Object.entries(runtime.metrics).map(([k, v]) => (
                <span key={k} className="metric-chip">
                  {k} <b>{Number.isInteger(v) ? v : v.toFixed(4)}</b>
                </span>
              ))}
            </div>
          </Section>
        )}

        <Section title="Logs">
          <LogTail logs={stageLogs} />
        </Section>

        {artifacts.length > 0 && (
          <Section title="Artifacts">
            <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 4 }}>
              {artifacts.map((a) => (
                <li key={a.name} className="mono" style={{ fontSize: 12, display: 'flex', justifyContent: 'space-between' }}>
                  <a href={artifactUrl(runId, a.name)} download>{a.name}</a>
                  <span className="faint">{(a.size_bytes / 1024).toFixed(1)} kB</span>
                </li>
              ))}
            </ul>
          </Section>
        )}

        <button onClick={onViewSpans} style={{ alignSelf: 'flex-start' }}>
          View spans for this stage →
        </button>
      </div>
    </aside>
  )
}
