import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it } from 'vitest'
import App from './App'

afterEach(cleanup)

describe('routing', () => {
  it('renders the landing page at /', () => {
    render(
      <MemoryRouter initialEntries={['/']}>
        <App />
      </MemoryRouter>,
    )
    expect(screen.getByTestId('landing-page')).toBeTruthy()
  })

  it('renders the studio (runs list) at /studio', () => {
    render(
      <MemoryRouter initialEntries={['/studio']}>
        <App />
      </MemoryRouter>,
    )
    // 'Runs' appears in both the rail nav and the page heading — assert at least one
    expect(screen.getAllByText('Runs').length).toBeGreaterThan(0)
  })
})
