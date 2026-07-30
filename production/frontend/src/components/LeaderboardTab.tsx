import { useEffect, useState } from 'react'
import { artifactUrl } from '../lib/api'
import type { ArtifactRow } from '../lib/types'

interface LeaderboardRow {
  config: string
  chunking?: string
  retrieval?: string
  top_k?: number
  recall: number
  precision: number
  precision_rw?: number
  f1: number
  f1_rw?: number
  winner?: boolean
}

// older runs' leaderboard.json predates the rank-weighted columns
const rankKey = (r: LeaderboardRow) => r.f1_rw ?? r.f1

interface Props {
  runId: string
  artifacts: ArtifactRow[]
}

export default function LeaderboardTab({ runId, artifacts }: Props) {
  const artifact = artifacts.find((a) => a.name === 'leaderboard.json')
  const [rows, setRows] = useState<LeaderboardRow[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!artifact) return
    fetch(artifactUrl(runId, artifact.name))
      .then((r) => r.json())
      .then((data) => setRows(data.rows ?? data))
      .catch((e) => setError(String(e)))
  }, [runId, artifact])

  if (!artifact)
    return (
      <div className="empty-state">
        The leaderboard appears after the auto-tune sweep evaluates each RAG configuration against the
        generated ground truth. Run the full pipeline to populate it.
      </div>
    )
  if (error) return <div className="empty-state mono">{error}</div>
  if (!rows) return <div className="empty-state">Loading leaderboard…</div>

  const best = rows.reduce((a, b) => (rankKey(b) > rankKey(a) ? b : a), rows[0])

  return (
    <div style={{ padding: 20, overflowY: 'auto' }}>
      <table className="data" style={{ maxWidth: 960 }}>
        <thead>
          <tr>
            <th>Config</th>
            <th>Chunking</th>
            <th>Retrieval</th>
            <th style={{ textAlign: 'right' }}>top-k</th>
            <th style={{ textAlign: 'right' }}>Recall</th>
            <th style={{ textAlign: 'right' }}>Prec (rw)</th>
            <th style={{ textAlign: 'right' }}>F1 (rw)</th>
            <th style={{ textAlign: 'right' }}>raw F1</th>
          </tr>
        </thead>
        <tbody>
          {[...rows]
            .sort((a, b) => rankKey(b) - rankKey(a))
            .map((r) => {
              const winner = r.winner ?? r === best
              return (
                <tr key={r.config} style={winner ? { background: 'color-mix(in srgb, var(--accent) 8%, transparent)' } : undefined}>
                  <td>{winner ? '★ ' : ''}{r.config}</td>
                  <td className="dim">{r.chunking ?? ''}</td>
                  <td className="dim">{r.retrieval ?? ''}</td>
                  <td style={{ textAlign: 'right' }}>{r.top_k ?? ''}</td>
                  <td style={{ textAlign: 'right' }}>{r.recall.toFixed(3)}</td>
                  <td style={{ textAlign: 'right' }}>{(r.precision_rw ?? r.precision).toFixed(3)}</td>
                  <td style={{ textAlign: 'right' }}><b>{rankKey(r).toFixed(3)}</b></td>
                  <td style={{ textAlign: 'right' }} className="dim">{r.f1.toFixed(3)}</td>
                </tr>
              )
            })}
        </tbody>
      </table>
    </div>
  )
}
