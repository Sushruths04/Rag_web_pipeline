import { STAGES, type Track } from './content'
import { useReveal } from './useReveal'

const TRACK_LABEL: Record<Track, string> = {
  gt: 'Track 1 — Ground truth',
  rag: 'Track 2 — RAG build',
  eval: 'Track 3 — Evaluation',
}

export default function PipelineSection() {
  const reveal = useReveal()
  return (
    <section id="how-it-works" className="section">
      <h2>One pipeline, three tracks, nineteen observable stages</h2>
      <p className="lede">
        Everything below runs live in the Studio with logs, spans, and metrics per stage.
        The GT track builds the ground truth; the RAG track builds the system under test
        from the same chunks; the eval track scores one against the other.
      </p>
      {(['gt', 'rag', 'eval'] as Track[]).map((track) => (
        <div key={track} className={`pipeline-track track-${track} reveal`} ref={reveal}>
          <div className="pipeline-track-label mono">{TRACK_LABEL[track]}</div>
          <div className="pipeline-row">
            {STAGES.filter((s) => s.track === track).map((s, i) => (
              <div key={s.name} className="stage-pill" style={{ transitionDelay: `${i * 70}ms` }} title={s.blurb}>
                <span className="stage-pill-label">{s.label}</span>
                <span className="stage-pill-blurb">{s.blurb}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </section>
  )
}
