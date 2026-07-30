import type { PipelineEvent } from './types'

export type WsStatus = 'live' | 'reconnecting' | 'closed'

/**
 * Subscribe to a run's event stream. Reconnects with backoff, resuming from
 * the last seen seq so no event is ever missed (server replays after=seq).
 */
export function subscribeRun(
  runId: string,
  afterSeq: number,
  onEvent: (ev: PipelineEvent) => void,
  onStatus: (s: WsStatus) => void,
): () => void {
  let lastSeq = afterSeq
  let ws: WebSocket | null = null
  let closed = false
  let retryMs = 500
  let timer: number | undefined

  const proto = location.protocol === 'https:' ? 'wss' : 'ws'

  function connect() {
    if (closed) return
    ws = new WebSocket(`${proto}://${location.host}/ws/runs/${runId}?after=${lastSeq}`)
    ws.onopen = () => {
      retryMs = 500
      onStatus('live')
    }
    ws.onmessage = (m) => {
      const ev = JSON.parse(m.data) as PipelineEvent
      if (ev.seq > lastSeq) {
        lastSeq = ev.seq
        onEvent(ev)
      }
    }
    ws.onclose = () => {
      if (closed) return
      onStatus('reconnecting')
      timer = window.setTimeout(connect, retryMs)
      retryMs = Math.min(retryMs * 2, 8000)
    }
    ws.onerror = () => ws?.close()
  }

  connect()
  return () => {
    closed = true
    window.clearTimeout(timer)
    ws?.close()
    onStatus('closed')
  }
}
