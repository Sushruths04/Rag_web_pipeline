import { describe, expect, it } from 'vitest'
import type { StageInfo } from './types'
import { COL_W, layoutTopology, ROW_H } from './layout'

// mirror of backend GRAFT_TOPOLOGY (names/deps/tracks)
const topo: StageInfo[] = [
  { name: 'profile', label: 'Structure Profile', deps: [], track: 'gt' },
  { name: 'ingest', label: 'Ingest & Parse', deps: ['profile'], track: 'gt' },
  { name: 'chunk', label: 'Adaptive Chunking', deps: ['ingest'], track: 'gt' },
  { name: 'extract', label: 'Fact Extraction', deps: ['chunk'], track: 'gt' },
  { name: 'clean', label: 'Fact Cleaning', deps: ['extract'], track: 'gt' },
  { name: 'graph', label: 'Bridge Graph Build', deps: ['clean'], track: 'gt' },
  { name: 'sample', label: 'Multi-hop Sampling', deps: ['graph'], track: 'gt' },
  { name: 'qagen', label: 'QA Generation (LLM)', deps: ['sample'], track: 'gt' },
  { name: 'verify', label: 'Verification', deps: ['qagen'], track: 'gt' },
  { name: 'gt_dataset', label: 'GT Dataset', deps: ['verify'], track: 'gt' },
  { name: 'bm25_index', label: 'BM25 Index', deps: ['chunk'], track: 'rag' },
  { name: 'vector_index', label: 'Vector Index', deps: ['chunk'], track: 'rag' },
  { name: 'fusion', label: 'Hybrid Fusion', deps: ['bm25_index', 'vector_index'], track: 'rag' },
  { name: 'rerank_warmup', label: 'Reranker Warm-up', deps: ['fusion'], track: 'rag' },
  { name: 'sweep', label: 'Auto-tune Sweep', deps: ['gt_dataset', 'fusion', 'rerank_warmup'], track: 'eval' },
  { name: 'leaderboard', label: 'Leaderboard', deps: ['sweep'], track: 'eval' },
  { name: 'select_config', label: 'Best-Config Selection', deps: ['leaderboard'], track: 'eval' },
  { name: 'final_eval', label: 'Final Eval Run', deps: ['select_config'], track: 'eval' },
  { name: 'report', label: 'Report', deps: ['final_eval'], track: 'eval' },
]

describe('layoutTopology', () => {
  const { nodes, edges } = layoutTopology(topo)
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]))

  it('positions all 19 nodes with finite coordinates', () => {
    expect(nodes.length).toBe(19)
    for (const n of nodes) {
      expect(Number.isFinite(n.position.x)).toBe(true)
      expect(Number.isFinite(n.position.y)).toBe(true)
    }
  })

  it('column = longest-path depth', () => {
    expect(byId.profile.position.x).toBe(0)
    expect(byId.chunk.position.x).toBe(2 * COL_W)
    // fusion depends on indexes at depth 3 -> depth 4
    expect(byId.fusion.position.x).toBe(4 * COL_W)
    // sweep waits for gt_dataset (depth 9) -> depth 10
    expect(byId.sweep.position.x).toBe(10 * COL_W)
  })

  it('parallel siblings in the same track+column stack vertically', () => {
    expect(byId.bm25_index.position.x).toBe(byId.vector_index.position.x)
    expect(byId.bm25_index.position.y).not.toBe(byId.vector_index.position.y)
  })

  it('tracks occupy separate row bands', () => {
    expect(byId.bm25_index.position.y).toBeGreaterThanOrEqual(ROW_H)
    expect(byId.sweep.position.y).toBeGreaterThanOrEqual(2 * ROW_H)
    expect(byId.ingest.position.y).toBeLessThan(ROW_H)
  })

  it('one edge per dependency', () => {
    expect(edges.length).toBe(topo.reduce((n, s) => n + s.deps.length, 0))
    expect(edges).toContainEqual(
      expect.objectContaining({ source: 'chunk', target: 'bm25_index' }),
    )
  })

  it('is deterministic', () => {
    expect(layoutTopology(topo)).toEqual(layoutTopology(topo))
  })

  it('carries label and track in node data', () => {
    expect(byId.qagen.data).toEqual({ label: 'QA Generation (LLM)', track: 'gt' })
  })
})
