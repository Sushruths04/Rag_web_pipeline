import type { StageInfo } from './types'

export const COL_W = 220
export const ROW_H = 150
const STACK_H = 96

export interface LayoutNode {
  id: string
  position: { x: number; y: number }
  data: { label: string; track: StageInfo['track'] }
}

export interface LayoutEdge {
  id: string
  source: string
  target: string
}

const TRACK_ROW: Record<StageInfo['track'], number> = { gt: 0, rag: 1, eval: 2 }

/**
 * Deterministic layered layout: column = longest-path depth from the roots,
 * row band = track (gt above, rag middle, eval below). Nodes sharing a
 * track+column stack vertically within the band.
 */
export function layoutTopology(stages: StageInfo[]): { nodes: LayoutNode[]; edges: LayoutEdge[] } {
  const depth: Record<string, number> = {}
  const byName = Object.fromEntries(stages.map((s) => [s.name, s]))

  const resolve = (name: string, seen: Set<string>): number => {
    if (name in depth) return depth[name]
    if (seen.has(name)) throw new Error(`cycle at ${name}`)
    seen.add(name)
    const s = byName[name]
    const d = s.deps.length === 0 ? 0 : Math.max(...s.deps.map((p) => resolve(p, seen))) + 1
    depth[name] = d
    return d
  }
  for (const s of stages) resolve(s.name, new Set())

  // stack order within (track, column): stable by topology array order
  const stackIndex: Record<string, number> = {}
  const counts: Record<string, number> = {}
  for (const s of stages) {
    const key = `${s.track}:${depth[s.name]}`
    stackIndex[s.name] = counts[key] ?? 0
    counts[key] = (counts[key] ?? 0) + 1
  }

  const nodes: LayoutNode[] = stages.map((s) => ({
    id: s.name,
    position: {
      x: depth[s.name] * COL_W,
      y: TRACK_ROW[s.track] * ROW_H + stackIndex[s.name] * STACK_H,
    },
    data: { label: s.label, track: s.track },
  }))

  const edges: LayoutEdge[] = stages.flatMap((s) =>
    s.deps.map((d) => ({ id: `${d}->${s.name}`, source: d, target: s.name })),
  )

  return { nodes, edges }
}
