import { COMPONENTS } from './content'
import { useReveal } from './useReveal'

export default function ComponentGrid() {
  const reveal = useReveal()
  return (
    <section id="components" className="section">
      <h2>Every component, explained</h2>
      <p className="lede">
        Click a card for the details. Cards marked <span className="cost-tag paid">LLM $</span> are
        the only places API money is ever spent — everything else runs locally.
      </p>
      <div className="component-grid reveal" ref={reveal}>
        {COMPONENTS.map((c, i) => (
          <details key={c.title} className={`component-card track-${c.track}`} style={{ transitionDelay: `${i * 45}ms` }}>
            <summary>
              <span className="component-title">{c.title}</span>
              <span className={`cost-tag ${c.costsMoney ? 'paid' : 'free'}`}>{c.costsMoney ? 'LLM $' : 'local'}</span>
              <span className="component-summary">{c.summary}</span>
            </summary>
            <ul>
              {c.details.map((d) => (
                <li key={d}>{d}</li>
              ))}
            </ul>
          </details>
        ))}
      </div>
    </section>
  )
}
