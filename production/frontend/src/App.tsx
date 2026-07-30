import { NavLink, Outlet, Route, Routes } from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import RunsPage from './pages/RunsPage'
import RunView from './pages/RunView'

function StudioShell() {
  return (
    <div className="app-shell">
      <aside className="rail">
        <div className="logo">
          GRAFT<span>·</span>Studio
        </div>
        <nav>
          <NavLink to="/studio" end className={({ isActive }) => (isActive ? 'active' : '')}>
            Runs
          </NavLink>
          <NavLink to="/" end>
            About GRAFT
          </NavLink>
        </nav>
        <div style={{ marginTop: 'auto' }} className="faint mono">
          grounded GT · LLM-free eval
        </div>
      </aside>
      <main className="main">
        <Outlet />
      </main>
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route element={<StudioShell />}>
        <Route path="/studio" element={<RunsPage />} />
        <Route path="/runs/:id" element={<RunView />} />
      </Route>
    </Routes>
  )
}
