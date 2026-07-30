import { Link } from 'react-router-dom'
import ComponentGrid from './landing/ComponentGrid'
import ConstellationCanvas from './landing/ConstellationCanvas'
import ModesSection from './landing/ModesSection'
import ObservabilitySection from './landing/ObservabilitySection'
import PipelineSection from './landing/PipelineSection'
import './landing/landing.css'

export default function LandingPage() {
  return (
    <div data-testid="landing-page" className="landing">
      <header className="hero">
        <nav className="landing-nav">
          <div className="logo">
            GRAFT<span>·</span>Studio
          </div>
          <div className="spacer" />
          <a className="navlink" href="#how-it-works">How it works</a>
          <a className="navlink" href="#components">Components</a>
          <a className="navlink" href="#modes">Free vs Live</a>
          <Link to="/studio"><button className="primary">Open Studio</button></Link>
        </nav>
        <ConstellationCanvas />
        <div className="hero-inner">
          <div className="hero-kicker">GRAFT · grounded RAG ground truth</div>
          <h1>
            Ground truth your retrieval can be <span className="ev">measured</span> against.
          </h1>
          <p className="sub">
            Upload PDFs. GRAFT extracts span-anchored facts, links them across pages into a
            verified fact graph, generates evidence-dense QA pairs — every hop grounded to
            page and bounding box — then auto-tunes and scores retrieval against them.
            No LLM judge anywhere in the scoring.
          </p>
          <div className="cta-row">
            <Link to="/studio"><button className="primary">Open Studio</button></Link>
            <a href="#how-it-works"><button>How it works</button></a>
          </div>
          <div className="hero-stats">
            <div className="hero-stat"><b>19</b>observable stages</div>
            <div className="hero-stat"><b>492</b>verified QA pairs</div>
            <div className="hero-stat"><b>$0</b>to evaluate</div>
            <div className="hero-stat"><b>◆ bbox</b>evidence grounding</div>
          </div>
        </div>
      </header>
      <PipelineSection />
      <ComponentGrid />
      <ModesSection />
      <ObservabilitySection />
      <footer className="landing-footer">
        GRAFT — evidence-grounded RAG evaluation ·
        deterministic scoring, auditable end to end
      </footer>
    </div>
  )
}
