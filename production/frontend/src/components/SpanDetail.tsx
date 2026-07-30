import type { Span } from '../lib/types'

function Json({ title, value }: { title: string; value: unknown }) {
  if (value == null) return null
  return (
    <details open>
      <summary className="dim" style={{ cursor: 'pointer', fontSize: 12, marginBottom: 4 }}>{title}</summary>
      <pre className="json">{JSON.stringify(value, null, 2)}</pre>
    </details>
  )
}

interface Props {
  span: Span
  parents: Span[]   // root-first chain of ancestors
  children: Span[]
  onSelect: (spanId: string) => void
}

export default function SpanDetail({ span, parents, children, onSelect }: Props) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, minWidth: 0 }}>
      {parents.length > 0 && (
        <div className="mono dim" style={{ fontSize: 11, display: 'flex', gap: 4, flexWrap: 'wrap', alignItems: 'center' }}>
          {parents.map((p) => (
            <span key={p.span_id}>
              <a onClick={() => onSelect(p.span_id)} style={{ cursor: 'pointer' }}>{p.name}</a>
              <span className="faint"> / </span>
            </span>
          ))}
          <b>{span.name}</b>
        </div>
      )}

      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <span className={`span-type ${span.type}`}>{span.type}</span>
        <b>{span.name}</b>
        {span.status === 'error' && <span className="badge-err">error</span>}
      </div>

      <div className="mono dim" style={{ fontSize: 11.5, display: 'flex', gap: 14, flexWrap: 'wrap' }}>
        <span>stage {span.stage}</span>
        {span.duration_ms != null && <span>{span.duration_ms.toFixed(1)} ms</span>}
        {span.tokens_in != null && <span>{span.tokens_in} tok in</span>}
        {span.tokens_out != null && <span>{span.tokens_out} tok out</span>}
        {span.cost_usd != null && <span>${span.cost_usd.toFixed(6)}</span>}
      </div>

      <Json title="Input" value={span.input} />
      <Json title="Output" value={span.output} />
      <Json title="Extra" value={span.extra} />

      {children.length > 0 && (
        <div>
          <div className="faint mono" style={{ fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>
            Children ({children.length})
          </div>
          <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 3 }}>
            {children.map((c) => (
              <li key={c.span_id} className="mono" style={{ fontSize: 12 }}>
                <a onClick={() => onSelect(c.span_id)} style={{ cursor: 'pointer' }}>
                  <span className={`span-type ${c.type}`} style={{ marginRight: 6 }}>{c.type}</span>
                  {c.name}
                </a>
                {c.duration_ms != null && <span className="faint"> · {c.duration_ms.toFixed(0)} ms</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
