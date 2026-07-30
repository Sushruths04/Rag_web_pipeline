import { useEffect, useRef } from 'react'

// The hero animation depicts the actual mechanism: facts cluster into page
// bands; every few seconds a verified bridge edge draws itself across two
// distant pages, pulses at both endpoints, and fades. Not a generic particle
// net — the drawing IS the pipeline's Stage "Bridge Graph Build".

interface Node {
  x: number
  y: number
  baseX: number
  vy: number
  r: number
  band: number
}

interface Bridge {
  from: number
  to: number
  start: number // timestamp ms
}

const GT = '23, 226, 182' // teal
const RAG = '77, 163, 255' // blue
const EVAL = '183, 140, 255' // violet
const BAND_COLORS = [GT, RAG, EVAL]
const N_BANDS = 5
const NODES_PER_BAND = 13
const BRIDGE_EVERY_MS = 2600
const BRIDGE_LIFE_MS = 2200

export default function ConstellationCanvas() {
  const ref = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    let raf = 0
    let nodes: Node[] = []
    let bridges: Bridge[] = []
    let lastBridge = 0

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      canvas.width = canvas.offsetWidth * dpr
      canvas.height = canvas.offsetHeight * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      const w = canvas.offsetWidth
      const h = canvas.offsetHeight
      nodes = []
      for (let b = 0; b < N_BANDS; b++) {
        // page bands spread across the width, each a loose vertical column
        const cx = ((b + 0.5) / N_BANDS) * w
        for (let i = 0; i < NODES_PER_BAND; i++) {
          const x = cx + (Math.random() - 0.5) * w * 0.11
          nodes.push({
            x,
            baseX: x,
            y: Math.random() * h,
            vy: 0.12 + Math.random() * 0.22,
            r: 1.1 + Math.random() * 1.7,
            band: b,
          })
        }
      }
      bridges = []
    }

    const spawnBridge = (t: number) => {
      // a bridge links two facts at least two pages apart — the whole point
      const a = Math.floor(Math.random() * nodes.length)
      const candidates = nodes
        .map((_, i) => i)
        .filter((i) => Math.abs(nodes[i].band - nodes[a].band) >= 2)
      if (!candidates.length) return
      const b = candidates[Math.floor(Math.random() * candidates.length)]
      bridges.push({ from: a, to: b, start: t })
      if (bridges.length > 3) bridges.shift()
    }

    const frame = (t: number) => {
      const w = canvas.offsetWidth
      const h = canvas.offsetHeight
      ctx.clearRect(0, 0, w, h)

      // page-band separators — faint verticals, the "pages"
      for (let b = 1; b < N_BANDS; b++) {
        const x = (b / N_BANDS) * w
        ctx.strokeStyle = 'rgba(34, 48, 41, 0.55)'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(x, h * 0.06)
        ctx.lineTo(x, h * 0.94)
        ctx.stroke()
      }

      // nodes drift upward inside their band, wrap at top
      for (const n of nodes) {
        if (!reduced) {
          n.y -= n.vy
          if (n.y < -4) n.y = h + 4
          n.x = n.baseX + Math.sin(t / 1800 + n.baseX) * 6
        }
        const color = BAND_COLORS[n.band % BAND_COLORS.length]
        const pulse = reduced ? 0.5 : 0.35 + 0.2 * Math.sin(t / 1100 + n.baseX)
        ctx.beginPath()
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(${color}, ${pulse})`
        ctx.fill()
      }

      // faint intra-page links between vertical neighbours
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          if (nodes[i].band !== nodes[j].band) continue
          const dy = nodes[i].y - nodes[j].y
          const dx = nodes[i].x - nodes[j].x
          const d = Math.hypot(dx, dy)
          if (d < 74) {
            ctx.beginPath()
            ctx.moveTo(nodes[i].x, nodes[i].y)
            ctx.lineTo(nodes[j].x, nodes[j].y)
            ctx.strokeStyle = `rgba(143, 166, 157, ${0.09 * (1 - d / 74)})`
            ctx.lineWidth = 1
            ctx.stroke()
          }
        }
      }

      // the signature: a verified bridge draws itself across distant pages
      if (!reduced && t - lastBridge > BRIDGE_EVERY_MS) {
        spawnBridge(t)
        lastBridge = t
      }
      for (const br of bridges) {
        const age = t - br.start
        if (age > BRIDGE_LIFE_MS) continue
        const a = nodes[br.from]
        const b = nodes[br.to]
        const draw = Math.min(1, age / 600) // edge draws in over 600ms
        const fade = age > 1400 ? 1 - (age - 1400) / (BRIDGE_LIFE_MS - 1400) : 1
        const mx = a.x + (b.x - a.x) * draw
        const my = a.y + (b.y - a.y) * draw
        ctx.beginPath()
        ctx.moveTo(a.x, a.y)
        ctx.lineTo(mx, my)
        ctx.strokeStyle = `rgba(${GT}, ${0.75 * fade})`
        ctx.lineWidth = 1.4
        ctx.stroke()
        // endpoint pulses: evidence located on both pages
        for (const p of draw >= 1 ? [a, b] : [a]) {
          ctx.beginPath()
          ctx.arc(p.x, p.y, 3.2 + 1.6 * Math.sin(age / 130), 0, Math.PI * 2)
          ctx.strokeStyle = `rgba(${GT}, ${0.5 * fade})`
          ctx.lineWidth = 1
          ctx.stroke()
        }
      }

      if (!reduced) raf = requestAnimationFrame(frame)
    }

    resize()
    if (reduced) {
      // static frame: bands + nodes + one frozen bridge for the idea
      spawnBridge(700)
      frame(700)
    } else {
      raf = requestAnimationFrame(frame)
    }
    window.addEventListener('resize', resize)
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return <canvas ref={ref} className="hero-canvas" aria-hidden="true" />
}
