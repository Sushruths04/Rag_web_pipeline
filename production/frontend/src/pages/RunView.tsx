import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import { Background, ReactFlow, type Edge, type Node } from '@xyflow/react'
import { getRun, getTopology } from '../lib/api'
import { layoutTopology } from '../lib/layout'
import { initRunState, runReducer, type RunState } from '../lib/runReducer'
import { subscribeRun, type WsStatus } from '../lib/ws'
import type { PipelineEvent, RunDetail, StageInfo } from '../lib/types'
import NodeInspector from '../components/NodeInspector'
import RunHeader from '../components/RunHeader'
import StageNode, { type StageNodeData } from '../components/StageNode'
import TracesTab from '../components/TracesTab'
import LeaderboardTab from '../components/LeaderboardTab'
import ReportTab from '../components/ReportTab'

const nodeTypes = { stage: StageNode }
const TABS = ['graph', 'traces', 'leaderboard', 'report'] as const
type Tab = (typeof TABS)[number]

export default function RunView() {
  const { id = '' } = useParams()
  const [params, setParams] = useSearchParams()
  const tab = (TABS.includes(params.get('tab') as Tab) ? params.get('tab') : 'graph') as Tab

  const [detail, setDetail] = useState<RunDetail | null>(null)
  const [topology, setTopology] = useState<StageInfo[]>([])
  const [run, setRun] = useState<RunState | null>(null)
  const [wsStatus, setWsStatus] = useState<WsStatus>('reconnecting')
  const [selected, setSelected] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const runRef = useRef<RunState | null>(null)
  runRef.current = run

  // initial snapshot + topology
  useEffect(() => {
    let live = true
    Promise.all([getRun(id), getTopology()])
      .then(([d, t]) => {
        if (!live) return
        setDetail(d)
        setTopology(t.stages)
        setRun(initRunState(d))
      })
      .catch((e) => setError(String(e)))
    return () => {
      live = false
    }
  }, [id])

  // live events
  const onEvent = useCallback((ev: PipelineEvent) => {
    setRun((s) => (s ? runReducer(s, ev) : s))
  }, [])

  useEffect(() => {
    if (!detail) return
    return subscribeRun(id, 0, onEvent, setWsStatus)
  }, [id, detail, onEvent])

  // refetch snapshot when a run reaches a terminal state (artifacts list, durations)
  const terminal = run && ['completed', 'failed', 'cancelled'].includes(run.status)
  useEffect(() => {
    if (!terminal) return
    getRun(id).then(setDetail).catch(() => undefined)
  }, [terminal, id])

  const base = useMemo(() => (topology.length ? layoutTopology(topology) : null), [topology])

  const nodes: Node<StageNodeData>[] = useMemo(() => {
    if (!base || !run) return []
    return base.nodes.map((n) => ({
      id: n.id,
      position: n.position,
      type: 'stage' as const,
      data: { ...n.data, runtime: run.stages[n.id] },
      selected: selected === n.id,
    }))
  }, [base, run, selected])

  const edges: Edge[] = useMemo(() => {
    if (!base || !run) return []
    return base.edges.map((e) => ({
      ...e,
      animated: run.stages[e.source]?.status === 'running',
    }))
  }, [base, run])

  if (error) return <div className="empty-state mono">{error}</div>
  if (!run || !detail || !base) return <div className="empty-state">Loading run…</div>

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <RunHeader run={run} createdAt={detail.created_at} wsStatus={wsStatus} />
      <div className="tabs">
        {TABS.map((t) => (
          <button key={t} className={tab === t ? 'active' : ''} onClick={() => setParams({ tab: t })}>
            {t[0].toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {tab === 'graph' && (
        <div style={{ flex: 1, position: 'relative', minHeight: 0 }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodeClick={(_, n) => setSelected(n.id)}
            onPaneClick={() => setSelected(null)}
            fitView
            proOptions={{ hideAttribution: true }}
            nodesDraggable={false}
            nodesConnectable={false}
            colorMode="dark"
            style={{ background: 'var(--bg)' }}
          >
            <Background gap={22} size={1} color="var(--line-soft)" />
          </ReactFlow>
          {selected && run.stages[selected] && (
            <NodeInspector
              runId={id}
              stage={selected}
              label={base.nodes.find((n) => n.id === selected)?.data.label ?? selected}
              runtime={run.stages[selected]}
              runStatus={run.status}
              logs={run.logs}
              artifacts={detail.artifacts.filter((a) => a.stage === selected)}
              onClose={() => setSelected(null)}
              onViewSpans={() => setParams({ tab: 'traces', stage: selected })}
            />
          )}
        </div>
      )}
      {tab === 'traces' && (
        <TracesTab runId={id} runStatus={run.status} initialStage={params.get('stage') ?? ''} />
      )}
      {tab === 'leaderboard' && <LeaderboardTab runId={id} artifacts={detail.artifacts} />}
      {tab === 'report' && <ReportTab runId={id} artifacts={detail.artifacts} />}
    </div>
  )
}
