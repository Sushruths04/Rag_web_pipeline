export type Track = 'gt' | 'rag' | 'eval'

export interface StageInfo {
  name: string
  label: string
  track: Track
  blurb: string
}

// Mirrors backend/app/stages/topology.py GRAFT_TOPOLOGY exactly (19 stages).
export const STAGES: StageInfo[] = [
  { name: 'profile', label: 'Structure Profile', track: 'gt', blurb: 'Profiles each PDF first — page count, scan detection, document type — and picks the right ingestion backend.' },
  { name: 'ingest', label: 'Ingest & Parse', track: 'gt', blurb: 'Parses the PDF into text units with page and character offsets preserved.' },
  { name: 'chunk', label: 'Adaptive Chunking', track: 'gt', blurb: 'Chunks each document with the strategy its profile recommends — the shared contract between GT and RAG tracks.' },
  { name: 'extract', label: 'Fact Extraction', track: 'gt', blurb: 'Extracts self-contained facts, each anchored to its source span (page, characters, bounding boxes).' },
  { name: 'clean', label: 'Fact Cleaning', track: 'gt', blurb: 'Drops fragments, boilerplate, and unanchored facts with per-reason counts — nothing vanishes silently.' },
  { name: 'graph', label: 'Bridge Graph Build', track: 'gt', blurb: 'Links facts that share verified bridge entities across pages. Deterministic — the LLM never invents a link.' },
  { name: 'sample', label: 'Multi-hop Sampling', track: 'gt', blurb: 'Samples single-hop neighbour pairs and cross-page bridge clusters from the verified graph.' },
  { name: 'qagen', label: 'QA Generation (LLM)', track: 'gt', blurb: 'Drafts answer-first questions over the sampled evidence — clauses bounded by their source facts.' },
  { name: 'verify', label: 'Verification', track: 'gt', blurb: 'NLI entailment per clause, single-fact sufficiency gates, and leave-one-out necessity — deterministic first, auditable always.' },
  { name: 'gt_dataset', label: 'GT Dataset', track: 'gt', blurb: 'Writes the verified QA set: every pair carries gold facts, chunks, pages, and bounding boxes.' },
  { name: 'bm25_index', label: 'BM25 Index', track: 'rag', blurb: 'Classic lexical index over the same chunks the GT was grounded in.' },
  { name: 'vector_index', label: 'Vector Index', track: 'rag', blurb: 'Dense sentence-embedding index built locally on CPU.' },
  { name: 'fusion', label: 'Hybrid Fusion', track: 'rag', blurb: 'Reciprocal-rank fusion of lexical and dense rankings.' },
  { name: 'rerank_warmup', label: 'Reranker Warm-up', track: 'rag', blurb: 'Loads and warms a cross-encoder reranker for the highest-precision configuration.' },
  { name: 'sweep', label: 'Auto-tune Sweep', track: 'eval', blurb: 'Scores every retrieval mode × top-k cell against the GT — recall, rank-weighted precision, F1.' },
  { name: 'leaderboard', label: 'Leaderboard', track: 'eval', blurb: 'Ranks all configurations; every number traces back to per-question evidence.' },
  { name: 'select_config', label: 'Best-Config Selection', track: 'eval', blurb: 'Picks the winner by rank-weighted F1 — smaller k wins ties.' },
  { name: 'final_eval', label: 'Final Eval Run', track: 'eval', blurb: 'Full-dataset evaluation at the winning configuration, with per-question retrieval spans.' },
  { name: 'report', label: 'Report', track: 'eval', blurb: 'Self-contained HTML report: headline metrics, per-hop breakdown, per-question evidence drill-down.' },
]

export interface ComponentInfo {
  title: string
  track: Track
  costsMoney: boolean
  summary: string
  details: string[]
}

export const COMPONENTS: ComponentInfo[] = [
  { title: 'Structure Profiler', track: 'gt', costsMoney: false, summary: 'Reads each PDF before anything else touches it.', details: ['Detects scanned vs digital pages, tables, and document type', 'Chooses the ingestion backend and chunking strategy per document', 'Zero API calls — pure local analysis'] },
  { title: 'Ingest & Adaptive Chunking', track: 'gt', costsMoney: false, summary: 'Parses and chunks with offsets preserved.', details: ['Every chunk keeps page + character provenance', 'The SAME chunks feed both the GT track and the RAG indexes — no train/test mismatch', 'Strategy is profile-driven, not one-size-fits-all'] },
  { title: 'Fact Extraction (SFU)', track: 'gt', costsMoney: true, summary: 'Extracts self-contained fact units from every chunk.', details: ['Each fact is span-anchored: page, character range, bounding boxes', 'Runs parallel LLM calls with a per-chunk resume cache', 'Rule-based fallback keeps the pipeline alive if a call fails'] },
  { title: 'Fact Cleaning', track: 'gt', costsMoney: false, summary: 'Adaptive filtering with per-reason drop counts.', details: ['Fragments, boilerplate, and front-matter are dropped at the source', 'Every drop is counted and reported — auditable, not silent'] },
  { title: 'Bridge Graph', track: 'gt', costsMoney: false, summary: 'Deterministic cross-page fact linking.', details: ['Facts sharing a verified entity (e.g. "ISO 15607") are linked across pages', 'The LLM never invents the multi-hop link — bridges are mined, then verified', 'Document-frequency banding rejects boilerplate pseudo-bridges'] },
  { title: 'QA Generation', track: 'gt', costsMoney: true, summary: 'Answer-first drafting over sampled evidence.', details: ['Single-hop: two neighbouring facts, question needs both', 'Multi-hop: 2+2 cross-page cluster, bridge entity hidden from the question', 'Clauses are source-bounded: no outside knowledge allowed in'] },
  { title: 'Verification (NLI + LOO)', track: 'gt', costsMoney: false, summary: 'Deterministic-first validation cascade.', details: ['Local NLI entailment per clause — no LLM judge', 'Single-fact sufficiency gate: no one fact may answer the whole question', 'Leave-one-out necessity: remove any fact and the answer must break'] },
  { title: 'BM25 / Vector / Hybrid Retrieval', track: 'rag', costsMoney: false, summary: 'Three retrieval families built locally.', details: ['Lexical BM25, dense sentence embeddings, reciprocal-rank fusion', 'All CPU-local: no embedding API costs', 'Chunk-agnostic scoring lets any chunking strategy compete fairly'] },
  { title: 'Cross-encoder Reranker', track: 'rag', costsMoney: false, summary: 'Highest-precision reranking stage.', details: ['Re-scores the hybrid candidates query-by-query', 'Local model, warmed up before the sweep so timings are honest'] },
  { title: 'Auto-tune Sweep + Leaderboard', track: 'eval', costsMoney: false, summary: 'Every config scored, one winner selected.', details: ['Modes × top-k grid, each cell scored on recall / rank-weighted precision / F1', 'Winner picked by rank-weighted F1; ties go to the smaller k', 'Per-question retrieval spans recorded for drill-down'] },
  { title: 'Evidence Report', track: 'eval', costsMoney: false, summary: 'Self-contained HTML you can hand to anyone.', details: ['Headline metrics with the method note: no LLM judge anywhere in scoring', 'Per-hop breakdown separates single-hop from multi-hop difficulty', 'Every question links to its gold evidence and what was actually retrieved'] },
]

export interface ModeInfo {
  title: string
  tag: string
  bullets: string[]
}

export const MODES: ModeInfo[] = [
  {
    title: 'Import mode',
    tag: 'FREE — $0.00',
    bullets: [
      'Uses the already-generated, already-verified GT dataset from an earlier paid run',
      'Zero API calls, zero cost — no LLM is contacted at all',
      'Still real end-to-end: parses your PDFs, chunks them, builds BM25 + vector + hybrid indexes, runs the sweep, scores retrieval — all with local models',
      'Limitation: it can only evaluate PDFs the imported GT already covers — a new PDF has no questions yet',
    ],
  },
  {
    title: 'Live mode',
    tag: 'PAID — budget-capped',
    bullets: [
      'Sends your PDF chunks to a real LLM API to extract facts and generate fresh questions and answers',
      'Costs real money per token — the max-cost budget cap aborts the run if exceeded, and max-pairs bounds the volume',
      'Use it when you upload a document the GT has never seen',
      'Fully traced: every LLM call becomes a span with tokens in/out and dollar cost, so you can audit exactly where the money went',
    ],
  },
]
