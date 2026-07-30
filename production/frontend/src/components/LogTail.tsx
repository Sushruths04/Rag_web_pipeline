import { useEffect, useRef, useState } from 'react'
import type { PipelineEvent } from '../lib/types'

export default function LogTail({ logs }: { logs: PipelineEvent[] }) {
  const box = useRef<HTMLDivElement>(null)
  const [follow, setFollow] = useState(true)

  useEffect(() => {
    if (follow && box.current) box.current.scrollTop = box.current.scrollHeight
  }, [logs, follow])

  if (logs.length === 0) return <div className="log-tail faint">No log lines yet.</div>

  return (
    <div
      className="log-tail"
      ref={box}
      onScroll={() => {
        const el = box.current
        if (!el) return
        setFollow(el.scrollHeight - el.scrollTop - el.clientHeight < 24)
      }}
    >
      {logs.map((ev) => {
        const level = String(ev.payload.level ?? 'info')
        return (
          <div key={ev.seq} className={level === 'warning' ? 'warn' : level === 'error' ? 'error' : ''}>
            <span className="faint">{new Date(ev.ts).toLocaleTimeString()} </span>
            {String(ev.payload.line ?? '')}
          </div>
        )
      })}
    </div>
  )
}
