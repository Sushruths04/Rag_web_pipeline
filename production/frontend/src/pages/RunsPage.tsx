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
  const [maxTokens, setMaxTokens] = useState('5000000')
  const [maxPairs, setMaxPairs] = useState('400')
  const [questionsPerDoc, setQuestionsPerDoc] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [trackCost, setTrackCost] = useState(false)
  const [priceIn, setPriceIn] = useState('')
  const [priceOut, setPriceOut] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
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
        config.max_pairs = Number(maxPairs) || 400
        if (questionsPerDoc.trim() !== '') {
          config.questions_per_doc = Number(questionsPerDoc)
        }
        if (llmMode === 'live') {
          // The API key is stripped from the persisted run config server-side
          // (see RunManager.split_secrets) and applied to the worker process
          // only. Leave it blank to use the server's configured key.
          if (apiKey.trim() !== '') config.api_key = apiKey.trim()
          // Tokens are the primary bound: the provider reports them, they need
          // no price table and they cannot go stale.
          config.max_tokens_total = Number(maxTokens) || 5_000_000
          // Cost estimation is opt-in. The provider's dashboard is the
          // authority; see docs/COST_TRACKING_IS_OPTIONAL.md.
          config.cost_tracking = trackCost ? 'estimate' : 'off'
          if (trackCost) {
            config.max_cost_usd = Number(maxCost) || 0
            if (priceIn.trim() !== '') config.price_in_per_mtok = Number(priceIn)
            if (priceOut.trim() !== '') config.price_out_per_mtok = Number(priceOut)
          }
        }
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
                  Reuse existing questions · free
                </button>
                <button type="button" className={llmMode === 'live' ? 'on' : ''} onClick={() => setLlmMode('live')}>
                  Generate new questions · uses your API key
                </button>
              </div>
              <ul className="mode-hints">
                {MODES[llmMode === 'import' ? 0 : 1].bullets.map((b) => (
                  <li key={b}>{b}</li>
                ))}
              </ul>
              {llmMode === 'live' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'end' }}>
                    <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
                      <span className="dim">questions per document</span>
                      <input
                        value={questionsPerDoc}
                        onChange={(e) => setQuestionsPerDoc(e.target.value)}
                        placeholder="all"
                        style={{ width: 120 }}
                      />
                    </label>
                    <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
                      <span className="dim">token limit</span>
                      <input value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} style={{ width: 110 }} />
                    </label>
                  </div>
                  <p className="dim" style={{ fontSize: 11, margin: 0 }}>
                    Questions are best effort — ask for 100 and a document that can only
                    support 20 will produce 20, and say so. Leave blank to take everything
                    each document supports. The token limit stops the run; your provider
                    meters what it actually costs.
                  </p>

                  <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
                    <span className="dim">API key (optional)</span>
                    <input
                      type="password"
                      autoComplete="off"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder="leave blank to use the server's configured key"
                      style={{ width: '100%', maxWidth: 460 }}
                    />
                  </label>
                  <p className="dim" style={{ fontSize: 11, margin: 0 }}>
                    Used for this run only. Never written to the run record or to any
                    artifact, log or trace span.
                  </p>

                  <button
                    type="button"
                    onClick={() => setShowAdvanced((v) => !v)}
                    style={{ alignSelf: 'start', fontSize: 12, background: 'none', border: 'none', textDecoration: 'underline', cursor: 'pointer', padding: 0 }}
                    className="dim"
                  >
                    {showAdvanced ? 'Hide' : 'Show'} advanced cost settings
                  </button>
                  {showAdvanced && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                      <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
                        <span className="dim">max pairs</span>
                        <input value={maxPairs} onChange={(e) => setMaxPairs(e.target.value)} style={{ width: 80 }} />
                      </label>

                      <label style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 12 }}>
                        <input
                          type="checkbox"
                          checked={trackCost}
                          onChange={(e) => setTrackCost(e.target.checked)}
                        />
                        <span>Also estimate cost in dollars (optional)</span>
                      </label>
                      <p className="dim" style={{ fontSize: 11, margin: 0 }}>
                        Off by default. Your provider meters spend and its dashboard is the
                        authority — anything shown here is an estimate from token counts and
                        a rate table that can go out of date. The run is bounded by the token
                        limit either way.
                      </p>
                      {trackCost && (
                        <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
                          <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
                            <span className="dim">stop at $ (estimate)</span>
                            <input value={maxCost} onChange={(e) => setMaxCost(e.target.value)} style={{ width: 90 }} />
                          </label>
                          <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
                            <span className="dim">$ / 1M input tok</span>
                            <input value={priceIn} onChange={(e) => setPriceIn(e.target.value)} placeholder="auto" style={{ width: 90 }} />
                          </label>
                          <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
                            <span className="dim">$ / 1M output tok</span>
                            <input value={priceOut} onChange={(e) => setPriceOut(e.target.value)} placeholder="auto" style={{ width: 90 }} />
                          </label>
                        </div>
                      )}
                    </div>
                  )}
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
