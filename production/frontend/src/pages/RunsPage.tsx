import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getRuns, startRun } from '../lib/api'
import type { RunSummary } from '../lib/types'
import StatusChip from '../components/StatusChip'
import UploadDropzone from '../components/UploadDropzone'
import { MODES } from './landing/content'

export default function RunsPage() {
  const nav = useNavigate()
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [files, setFiles] = useState<File[]>([])
  const [pipeline, setPipeline] = useState('graft')
  const [sleepScale, setSleepScale] = useState('0.05')
  const [llmMode, setLlmMode] = useState<'import' | 'live'>('import')
  const [maxCost, setMaxCost] = useState('5.0')
  const [maxPairs, setMaxPairs] = useState('400')
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(() => {
    getRuns()
      .then((r) => setRuns(r.runs))
      .catch((e) => setError(String(e)))
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  // keep the list fresh while anything is running
  useEffect(() => {
    if (!runs.some((r) => r.status === 'running' || r.status === 'queued')) return
    const t = setInterval(refresh, 5000)
    return () => clearInterval(t)
  }, [runs, refresh])

  const onStart = async () => {
    setStarting(true)
    setError(null)
    try {
      const config: Record<string, unknown> = {}
      if (pipeline === 'dummy') config.sleep_scale = Number(sleepScale) || 0.05
      if (pipeline === 'graft') {
        config.llm_mode = llmMode
        config.max_cost_usd = Number(maxCost) || 5.0
        config.max_pairs = Number(maxPairs) || 400
      }
      const { run_id } = await startRun(files, config, pipeline)
      nav(`/runs/${run_id}`)
    } catch (e) {
      setError(String(e))
      setStarting(false)
    }
  }

  return (
    <div style={{ padding: 28, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 24, maxWidth: 900 }}>
      <div>
        <h1 style={{ fontSize: 20, fontWeight: 700 }}>Runs</h1>
        <p className="dim" style={{ marginTop: 4, fontSize: 13 }}>
          Upload PDFs, generate grounded ground truth, and evaluate retrieval against it — no LLM judge.
        </p>
      </div>

      <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <UploadDropzone files={files} onFiles={setFiles} />
        <div style={{ display: 'flex', gap: 14, alignItems: 'end', flexWrap: 'wrap' }}>
          <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
            <span className="dim">Pipeline</span>
            <select value={pipeline} onChange={(e) => setPipeline(e.target.value)}>
              <option value="graft">graft (real pipeline)</option>
              <option value="dummy">dummy (simulated)</option>
            </select>
          </label>
          {pipeline === 'dummy' && (
            <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
              <span className="dim">sleep_scale</span>
              <input value={sleepScale} onChange={(e) => setSleepScale(e.target.value)} style={{ width: 90 }} />
            </label>
          )}
          {pipeline === 'graft' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, width: '100%' }}>
              <div className="segmented">
                <button type="button" className={llmMode === 'import' ? 'on' : ''} onClick={() => setLlmMode('import')}>
                  Free · import verified GT
                </button>
                <button type="button" className={llmMode === 'live' ? 'on' : ''} onClick={() => setLlmMode('live')}>
                  Live · fresh GT (paid)
                </button>
              </div>
              <ul className="mode-hints">
                {MODES[llmMode === 'import' ? 0 : 1].bullets.map((b) => (
                  <li key={b}>{b}</li>
                ))}
              </ul>
              {llmMode === 'live' && (
                <div style={{ display: 'flex', gap: 14 }}>
                  <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
                    <span className="dim">max cost $</span>
                    <input value={maxCost} onChange={(e) => setMaxCost(e.target.value)} style={{ width: 70 }} />
                  </label>
                  <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
                    <span className="dim">max pairs</span>
                    <input value={maxPairs} onChange={(e) => setMaxPairs(e.target.value)} style={{ width: 70 }} />
                  </label>
                </div>
              )}
            </div>
          )}
          <button className="primary" onClick={onStart} disabled={starting || (pipeline === 'graft' && files.length === 0)}>
            {starting ? 'Starting…' : 'Start run'}
          </button>
        </div>
        {error && <div style={{ color: 'var(--err)', fontSize: 12 }} className="mono">{error}</div>}
      </div>

      <div className="panel" style={{ padding: 0 }}>
        {runs.length === 0 ? (
          <div className="empty-state">No runs yet. Start one above to see the pipeline live.</div>
        ) : (
          <table className="data">
            <thead>
              <tr>
                <th>Run</th>
                <th>Pipeline</th>
                <th>Status</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id} className="clickable" onClick={() => nav(`/runs/${r.run_id}`)}>
                  <td>{r.run_id}</td>
                  <td className="dim">{r.pipeline}</td>
                  <td><StatusChip status={r.status} /></td>
                  <td className="dim">{new Date(r.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
