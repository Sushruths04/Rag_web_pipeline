import type { ArtifactRow, PipelineEvent, RunDetail, RunSummary, Span, StageInfo } from './types'

async function get<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${url}`)
  return r.json() as Promise<T>
}

async function post<T>(url: string, body?: FormData): Promise<T> {
  const r = await fetch(url, { method: 'POST', body })
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${url}`)
  return r.json() as Promise<T>
}

export const getTopology = () => get<{ stages: StageInfo[] }>('/api/topology')
export const getRuns = () => get<{ runs: RunSummary[] }>('/api/runs')
export const getRun = (id: string) => get<RunDetail>(`/api/runs/${id}`)

export const getEvents = (id: string, params: Record<string, string | number> = {}) => {
  const qs = new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)]))
  return get<{ events: PipelineEvent[] }>(`/api/runs/${id}/events?${qs}`)
}

export const getSpans = (id: string, params: Record<string, string | number> = {}) => {
  const qs = new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)]))
  return get<{ spans: Span[] }>(`/api/runs/${id}/spans?${qs}`)
}

export const getArtifacts = (id: string) => get<{ artifacts: ArtifactRow[] }>(`/api/runs/${id}/artifacts`)

export function startRun(files: File[], config: Record<string, unknown>, pipeline = 'dummy') {
  const fd = new FormData()
  for (const f of files) fd.append('files', f)
  fd.append('config', JSON.stringify(config))
  fd.append('pipeline', pipeline)
  return post<{ run_id: string }>('/api/runs', fd)
}

export const cancelRun = (id: string) => post<{ status: string }>(`/api/runs/${id}/cancel`)
export const retryStage = (id: string, stage: string) =>
  post<{ status: string }>(`/api/runs/${id}/stages/${stage}/retry`)

export const artifactUrl = (id: string, name: string) => `/api/runs/${id}/artifacts/${name}`
export const tracesExportUrl = (id: string) => `/api/runs/${id}/traces/export`
