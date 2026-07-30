import { MODES } from './content'
import { useReveal } from './useReveal'

export default function ModesSection() {
  const reveal = useReveal()
  return (
    <section id="modes" className="section">
      <h2>Two ways to run: free import, budget-capped live</h2>
      <p className="lede">
        The QA-generation stage is the only paid step in the whole pipeline — so it is
        the only step with a switch.
      </p>
      <div className="modes-row reveal" ref={reveal}>
        {MODES.map((m) => (
          <div key={m.title} className={`mode-card ${m.tag.startsWith('FREE') ? 'free' : 'paid'}`}>
            <div className="mode-tag mono">{m.tag}</div>
            <h3>{m.title}</h3>
            <ul>
              {m.bullets.map((b) => (
                <li key={b}>{b}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </section>
  )
}
