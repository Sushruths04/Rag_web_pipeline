import { useEffect, useMemo, useState } from 'react'
import { getSpans, tracesExportUrl } from '../lib/api'
import type { RunStatus, Span } from '../lib/types'
import SpanDetail from './SpanDetail'

interface Props {
  runId: string
  runStatus: RunStatus
  initialStage: string
}

const TYPES = ['', 'llm', 'retrieval', 'processing', 'embedding']

export default function TracesTab({ runId, runStatus, initialStage }: Props) {
  const [spans, setSpans] = useState<Span[]>([])
  const [loaded, setLoaded] = useState(false)
  const [stage, setStage] = useState(initialStage)
  const [type, setType] = useState('')
  const [search, setSearch] = useState('')
  const [minMs, setMinMs] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const terminal = ['completed', 'failed', 'cancelled'].includes(runStatus)

  useEffect(() => {
    let live = true
    getSpans(runId, { limit: 5000 }).then((r) => {
      if (!live) return
      setSpans(r.spans)
      setLoaded(true)
    })
    return () => {
      live = false
    }
  }, [runId, terminal])

  const byId = useMemo(() => Object.fromEntries(spans.map((s) => [s.span_id, s])), [spans])
  const childrenOf = useMemo(() => {
    const m: Record<string, Span[]> = {}
    for (const s of spans) {
      if (s.parent_id) (m[s.parent_id] ??= []).push(s)
    }
    return m
  }, [spans])

  const stages = useMemo(() => [...new Set(spans.map((s) => s.stage))], [spans])

  const filtered = useMemo(() => {
    const q = search.toLowerCase()
    const min = Number(minMs) || 0
    return spans.filter((s) => {
      if (stage && s.stage !== stage) return false
      if (type && s.type !== type) return false
      if (min && (s.duration_ms ?? 0) < min) return false
      if (q) {
        const hay = `${s.name} ${JSON.stringify(s.input ?? '')} ${JSON.stringify(s.output ?? '')}`.toLowerCase()
        if (!hay.includes(q)) return false
      }
      return true
    })
  }, [spans, stage, type, search, minMs])

  const totals = useMemo(
    () => ({
      cost: filtered.reduce((a, s) => a + (s.cost_usd ?? 0), 0),
      tokIn: filtered.reduce((a, s) => a + (s.tokens_in ?? 0), 0),
      tokOut: filtered.reduce((a, s) => a + (s.tokens_out ?? 0), 0),
    }),
    [filtered],
  )

  const selected = selectedId ? byId[selectedId] : null
  const parents = useMemo(() => {
    if (!selected) return []
    const chain: Span[] = []
    let p = selected.parent_id ? byId[selected.parent_id] : undefined
    while (p) {
      chain.unshift(p)
      p = p.parent_id ? byId[p.parent_id] : undefined
    }
    return chain
  }, [selected, byId])

  if (!loaded) return <div className="empty-state">Loading traces…</div>
  if (spans.length === 0)
    return <div className="empty-state">No spans recorded yet. They appear as soon as stages start doing work.</div>

  return (
    <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', gap: 8, padding: '10px 16px', flexWrap: 'wrap', alignItems: 'center' }}>
          <select value={stage} onChange={(e) => setStage(e.target.value)} aria-label="Filter by stage">
            <option value="">all stages</option>
            {stages.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <select value={type} onChange={(e) => setType(e.target.value)} aria-label="Filter by type">
            {TYPES.map((t) => (
              <option key={t} value={t}>{t || 'all types'}</option>
            ))}
          </select>
          <input placeholder="search name / payloads" value={search} onChange={(e) => setSearch(e.target.value)} style={{ width: 200 }} />
          <input placeholder="min ms" value={minMs} onChange={(e) => setMinMs(e.target.value)} style={{ width: 80 }} />
          <span className="mono dim" style={{ marginLeft: 'auto', fontSize: 11.5 }}>
            {filtered.length} spans · Σ ${totals.cost.toFixed(4)} · {totals.tokIn}/{totals.tokOut} tok
          </span>
          <a href={tracesExportUrl(runId)} download>
            <button>Export JSON</button>
          </a>
        </div>
        <div style={{ flex: 1, overflowY: 'auto' }}>
          <table className="data">
            <thead>
              <tr>
                <th>Time</th>
                <th>Stage</th>
                <th>Type</th>
                <th>Name</th>
                <th style={{ textAlign: 'right' }}>ms</th>
                <th style={{ textAlign: 'right' }}>tok</th>
                <th style={{ textAlign: 'right' }}>cost</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((s) => (
                <tr
                  key={s.span_id}
                  className="clickable"
                  onClick={() => setSelectedId(s.span_id)}
                  style={selectedId === s.span_id ? { background: 'var(--raised)' } : undefined}
                >
                  <td className="faint">{new Date(s.started_at).toLocaleTimeString()}</td>
                  <td>{s.stage}</td>
                  <td><span className={`span-type ${s.type}`}>{s.type}</span></td>
                  <td style={{ color: s.status === 'error' ? 'var(--err)' : undefined }}>{s.name}</td>
                  <td style={{ textAlign: 'right' }}>{s.duration_ms?.toFixed(0) ?? ''}</td>
                  <td style={{ textAlign: 'right' }} className="dim">
                    {s.tokens_in != null ? `${s.tokens_in}/${s.tokens_out ?? 0}` : ''}
                  </td>
                  <td style={{ textAlign: 'right' }}>{s.cost_usd != null ? `$${s.cost_usd.toFixed(6)}` : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      {selected && (
        <div style={{ width: 440, borderLeft: '1px solid var(--line-soft)', overflowY: 'auto', padding: 16 }}>
          <SpanDetail
            span={selected}
            parents={parents}
            children={childrenOf[selected.span_id] ?? []}
            onSelect={setSelectedId}
          />
        </div>
      )}
    </div>
  )
}
