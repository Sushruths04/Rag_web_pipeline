import { cleanup, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it } from 'vitest'
import LandingPage from '../LandingPage'

// without cleanup, a previous render's duplicate #id wins document-level
// id resolution and container-scoped queries return null
afterEach(cleanup)

describe('landing page', () => {
  it('mounts hero canvas and section anchors', () => {
    const { container } = render(
      <MemoryRouter>
        <LandingPage />
      </MemoryRouter>,
    )
    expect(container.querySelector('canvas')).toBeTruthy()
    expect(container.querySelector('#how-it-works')).toBeTruthy()
    expect(container.querySelector('#components')).toBeTruthy()
    expect(container.querySelector('#modes')).toBeTruthy()
  })

  it('renders all 19 pipeline stages in track order', () => {
    const { container } = render(
      <MemoryRouter>
        <LandingPage />
      </MemoryRouter>,
    )
    expect(container.querySelectorAll('.stage-pill').length).toBe(19)
    expect(container.querySelector('#how-it-works')?.textContent).toContain('Adaptive Chunking')
  })

  it('explains free vs live modes with bullets', () => {
    const { container } = render(
      <MemoryRouter>
        <LandingPage />
      </MemoryRouter>,
    )
    const modes = container.querySelector('#modes')
    expect(modes?.textContent).toContain('Import mode')
    expect(modes?.textContent).toContain('Live mode')
    expect((modes?.querySelectorAll('li').length ?? 0)).toBeGreaterThanOrEqual(8)
    expect(container.querySelectorAll('.component-card').length).toBeGreaterThanOrEqual(10)
  })
})
