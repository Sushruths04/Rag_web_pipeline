import { useReveal } from './useReveal'

export default function ObservabilitySection() {
  const reveal = useReveal()
  return (
    <section id="observability" className="section">
      <h2>No LLM judge. Every number has receipts.</h2>
      <div className="obs-grid reveal" ref={reveal}>
        <div className="obs-card">
          <h3>Deterministic scoring</h3>
          <p>
            A retrieved chunk is relevant when it contains at least 60% of a gold evidence
            unit's tokens. Rank-weighted precision credits putting the evidence at the top.
            Same inputs, same score, every time — reproducible by construction.
          </p>
        </div>
        <div className="obs-card">
          <h3>Grounded to the pixel</h3>
          <p>
            Every QA pair carries its gold facts <span className="ev-mark">◆</span>, chunks,
            pages, and bounding boxes, traced from extraction spans. You can open the PDF
            and point at the evidence.
          </p>
        </div>
        <div className="obs-card">
          <h3>Live spans for everything</h3>
          <p>
            Processing, embedding, retrieval, and LLM calls all emit spans — query,
            candidates, scores, verdicts, tokens, and dollars — streamed into the Studio
            while the run is still going.
          </p>
        </div>
      </div>
    </section>
  )
}
